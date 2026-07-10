"""
Агент-Критик на базе GigaChat.

Отвечает за:
    1. Проверку сцены на логические несоответствия
    2. Сверку с онтологической памятью (хронология, персонажи, улики)
    3. Возврат [APPROVE] если всё корректно, или списка замечаний
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)

# GigaChat API эндпоинты
_OAUTH_URL = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
_CHAT_URL = "https://gigachat.devices.sberbank.ru/api/v1/chat/completions"

APPROVE_MARKER = "[APPROVE]"

# ─────────────────────────── Системный промпт ─────────────────────────────────

_SYSTEM_CRITIC = """Ты — строгий литературный редактор и логик. Твоя задача —
проверить сцену детективного рассказа на:

1. Логические противоречия (персонаж не может быть в двух местах одновременно)
2. Несоответствие временной линии (past/present/future из памяти)
3. Противоречия с уже установленными фактами о персонажах
4. Улики, которые появляются «из ниоткуда» без подготовки
5. Нарушения причинно-следственных связей
6. Несоответствие месту/времени действия

ОБЯЗАТЕЛЬНЫЙ ФОРМАТ ОТВЕТА:

Если сцена логически корректна — ответь РОВНО ОДНОЙ строкой:
[APPROVE]

Если есть проблемы — перечисли их пронумерованным списком:
1. [описание проблемы]
2. [описание проблемы]
...

НЕ добавляй ничего лишнего. Только [APPROVE] или нумерованный список проблем.
Будь конкретен: укажи, какой факт из памяти противоречит тексту сцены.
"""


# ─────────────────────────── Агент-критик ────────────────────────────────────

class CriticAgent:
    """
    Агент-Критик на базе GigaChat.

    Верифицирует логическую согласованность сцен с онтологической памятью.
    """

    def __init__(
        self,
        credentials: str,
        scope: str = "GIGACHAT_API_PERS",
        model: str = "GigaChat",
    ):
        """
        Args:
            credentials : Base64-строка авторизации (GIGACHAT_CREDENTIALS из .env)
            scope       : OAuth scope
            model       : Имя модели GigaChat
        """
        self.credentials = credentials
        self.scope = scope
        self.model = model

        self._token: Optional[str] = None
        self._token_expires_at: float = 0.0

        # Статистика
        self.total_requests: int = 0
        self.total_approvals: int = 0
        self.total_rejections: int = 0

    # ── Авторизация ────────────────────────────────────────────────────────────

    def _ensure_token(self) -> str:
        """
        Получить (или обновить) OAuth-токен GigaChat.

        Токен кешируется и обновляется за 60 секунд до истечения.
        """
        now = time.time()
        if self._token and now < self._token_expires_at - 60:
            return self._token

        logger.debug("🔑 Critic: Получаю токен GigaChat...")
        response = requests.post(
            _OAUTH_URL,
            headers={
                "Authorization": f"Basic {self.credentials}",
                "RqUID": str(uuid.uuid4()),
                "Content-Type": "application/x-www-form-urlencoded",
                "Accept": "application/json",
            },
            data={"scope": self.scope},
            verify=False,
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        self._token = data["access_token"]
        # GigaChat токен живёт 30 минут (1800 сек)
        self._token_expires_at = now + 1800
        logger.debug("🔑 Critic: Токен получен")
        return self._token

    # ── API запрос ─────────────────────────────────────────────────────────────

    def _call_api(self, user_prompt: str) -> Tuple[str, int]:
        """
        Выполнить запрос к GigaChat API.

        Returns:
            Tuple[ответ модели, использовано токенов]
        """
        token = self._ensure_token()
        start_time = time.time()

        response = requests.post(
            _CHAT_URL,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": _SYSTEM_CRITIC},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.1,  # Низкая для детерминированной логической проверки
                "max_tokens": 1500,
                "stream": False,
            },
            verify=False,
            timeout=90,
        )
        response.raise_for_status()

        data = response.json()
        content = data["choices"][0]["message"]["content"].strip()
        tokens = data.get("usage", {}).get("total_tokens", 0)
        elapsed = time.time() - start_time

        self.total_requests += 1

        logger.debug(
            "🔍 Critic API: %.1f сек, %d токенов", elapsed, tokens
        )

        return content, tokens

    # ── Публичный API ──────────────────────────────────────────────────────────

    def review_scene(
        self,
        scene_text: str,
        memory_context: Dict[str, Any],
        scene_plan: Dict[str, Any],
        approved_scenes: List[Dict[str, Any]],
    ) -> Tuple[bool, str]:
        """
        Проверить сцену на логическую согласованность.

        Args:
            scene_text      : Текст черновика сцены от Писателя
            memory_context  : Полный контекст памяти из MCP
            scene_plan      : Ожидаемый план текущей сцены
            approved_scenes : Уже одобренные сцены (для проверки связности)

        Returns:
            Tuple[approved (bool), feedback (str)]
            - approved=True  → сцена логически корректна
            - approved=False → feedback содержит список замечаний
        """
        logger.info(
            "🔍 Critic: Проверяю сцену '%s'...",
            scene_plan.get("title", "?"),
        )

        import json

        # Формируем промпт для Критика
        # Берём только релевантные части памяти (не перегружаем контекст)
        memory_summary = _build_memory_summary(memory_context)
        previous_summary = _build_previous_scenes_summary(approved_scenes)

        user_prompt = (
            f"ПЛАН СЦЕНЫ (что должно произойти):\n"
            f"Цель: {scene_plan.get('goal', '')}\n"
            f"Ключевое событие: {scene_plan.get('key_event', '')}\n"
            f"Место: {scene_plan.get('location', '')}\n\n"
            f"ОНТОЛОГИЧЕСКАЯ ПАМЯТЬ (факты о мире):\n"
            f"{memory_summary}\n\n"
            + (f"ПРЕДЫДУЩИЕ СЦЕНЫ (краткое содержание):\n{previous_summary}\n\n"
               if previous_summary else "")
            + f"ТЕКСТ СЦЕНЫ ДЛЯ ПРОВЕРКИ:\n{scene_text}\n\n"
            f"Проверь сцену на логические несоответствия с памятью и предыдущими сценами."
        )

        content, tokens = self._call_api(user_prompt)

        # Определяем результат
        is_approved = content.strip().startswith(APPROVE_MARKER)

        if is_approved:
            self.total_approvals += 1
            logger.info("✅ Critic: [APPROVE] — сцена одобрена (%d токенов)", tokens)
        else:
            self.total_rejections += 1
            logger.info(
                "❌ Critic: Замечания найдены (%d токенов):\n%s",
                tokens, content[:300],
            )

        return is_approved, content

    def get_stats(self) -> Dict[str, int]:
        """Вернуть статистику проверок."""
        return {
            "total_requests": self.total_requests,
            "total_approvals": self.total_approvals,
            "total_rejections": self.total_rejections,
        }


# ─────────────────────────── Вспомогательные функции ──────────────────────────

def _build_memory_summary(memory_context: Dict[str, Any]) -> str:
    """
    Создать компактное текстовое резюме памяти для Критика.

    Извлекает только ключевые факты, чтобы не перегружать контекст.
    """
    import json

    lines: List[str] = []

    for layer in ["past", "present", "future"]:
        if layer not in memory_context:
            continue
        layer_data = memory_context[layer]
        if not isinstance(layer_data, dict):
            continue

        lines.append(f"\n[{layer.upper()}]")

        # Состояние мира
        world = layer_data.get("world_state") or {}
        if world:
            lines.append(f"Мир: {json.dumps(world, ensure_ascii=False)}")

        # Персонажи
        chars = layer_data.get("characters") or {}
        if chars:
            for name, info in chars.items():
                if isinstance(info, dict):
                    loc = info.get("location", "неизвестно")
                    status = info.get("status", "")
                    role = info.get("role", "")
                    lines.append(f"  {name} ({role}): {status}, в {loc}")

        # Улики
        clues = layer_data.get("clues") or []
        if clues:
            lines.append("Улики:")
            for clue in clues:
                if isinstance(clue, dict):
                    lines.append(
                        f"  - [{clue.get('id', '?')}] {clue.get('description', '')} "
                        f"(статус: {clue.get('status', 'unknown')})"
                    )

        # Последние события
        events = layer_data.get("events") or []
        if events:
            lines.append("События:")
            for event in events[-3:]:  # Только последние 3
                if isinstance(event, dict):
                    lines.append(f"  - {event.get('description', '')}")

    return "\n".join(lines) if lines else "Память пуста"


def _build_previous_scenes_summary(approved_scenes: List[Dict[str, Any]]) -> str:
    """Создать краткое резюме предыдущих сцен для контекста Критика."""
    if not approved_scenes:
        return ""
    summaries = []
    for scene in approved_scenes[-3:]:  # Последние 3 сцены
        title = scene.get("title", f"Сцена {scene.get('index', '?')}")
        text = scene.get("text", "")
        # Берём только первые 200 символов каждой сцены
        preview = text[:200] + "..." if len(text) > 200 else text
        summaries.append(f"[{title}]: {preview}")
    return "\n".join(summaries)