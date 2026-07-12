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

_SYSTEM_CRITIC = """Ты — строгий литературный редактор и логик. Проверяй сцену
детективного рассказа по следующим критериям:

ПРИОРИТЕТ 1 — ДЛИНА:
- Сцена должна быть 300–550 слов (считай слова сам)
- Короче 300 → отклони: "Сцена слишком короткая (X слов)"
- Длиннее 550 → отклони: "Сцена слишком длинная (X слов)"

ПРИОРИТЕТ 2 — КАНОНИЧЕСКИЕ ФАКТЫ (раздел "КАНОНИЧЕСКИЕ ФАКТЫ" в промпте):
- Имена и родство персонажей НЕЛЬЗЯ менять (сын ≠ племянник)
- Роли персонажей НЕЛЬЗЯ менять
- Число подозреваемых = все персонажи КРОМЕ следователя/детектива
- Погода/время суток зафиксированы — не должны меняться
- Если сцена противоречит каноническому факту → ОБЯЗАТЕЛЬНО отклони

ПРИОРИТЕТ 3 — ЛОГИКА:
- Персонаж не может быть в двух местах одновременно
- Мёртвый персонаж не может действовать
- Улики должны появляться логично, не "из воздуха"
- Причинно-следственные связи должны соблюдаться

ОБЯЗАТЕЛЬНЫЙ ФОРМАТ ОТВЕТА:
Если всё корректно → одна строка: [APPROVE]
Если есть проблемы → нумерованный список:
1. [конкретная проблема со ссылкой на канонический факт]
2. ...

ВАЖНО: будь строгим. Любое противоречие с каноническими фактами = отклонение.
Не одобряй сцену если есть хоть одно реальное противоречие.
"""


class CriticAgent:
    def __init__(
        self,
        credentials: str,
        scope: str = "GIGACHAT_API_PERS",
        model: str = "GigaChat",
    ):
        self.credentials = credentials
        self.scope = scope
        self.model = model

        self._token: Optional[str] = None
        self._token_expires_at: float = 0.0

        # ← Исправлено: единое написание без подчёркивания
        self.total_requests: int = 0
        self.total_approvals: int = 0
        self.total_rejections: int = 0

    def _ensure_token(self) -> str:
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
        self._token_expires_at = now + 1800
        return self._token

    def _call_api(self, user_prompt: str) -> Tuple[str, int]:
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
                "temperature": 0.05,  # Почти детерминированный
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
        logger.debug("🔍 Critic API: %.1f сек, %d токенов", elapsed, tokens)

        return content, tokens

    def review_scene(
        self,
        scene_text: str,
        memory_context: Dict[str, Any],
        scene_plan: Dict[str, Any],
        approved_scenes: List[Dict[str, Any]],
        story_bible: str = "",          # ← новый параметр
        tracker_conflicts: Optional[List[Dict[str, Any]]] = None,  # ← из StateTracker
    ) -> Tuple[bool, str]:

        word_count = len(scene_text.split())

        if word_count > 600:
            self.total_rejections += 1
            return False, f"Сцена слишком длинная: {word_count} слов. Сократи до 300–550."

        if word_count < 250:
            self.total_rejections += 1
            return False, f"Сцена слишком короткая: {word_count} слов. Минимум 300."

        # Если StateTracker уже нашёл критические конфликты — сразу отклоняем
        if tracker_conflicts:
            critical = [c for c in tracker_conflicts if c.get("severity") == "critical"]
            if critical:
                feedback_lines = ["StateTracker обнаружил критические конфликты:"]
                for c in critical:
                    feedback_lines.append(
                        f"{len(feedback_lines)}. {c.get('description', '')} "
                        f"(канон: '{c.get('canonical_fact', '')}', "
                        f"в тексте: '{c.get('scene_fact', '')}')"
                    )
                self.total_rejections += 1
                return False, "\n".join(feedback_lines)

        canonical = _build_canonical_facts(memory_context)
        memory_summary = _build_memory_summary(memory_context)
        previous_summary = _build_previous_scenes_summary(approved_scenes)

        user_prompt = (
            # Story Bible — первый блок
            (f"STORY BIBLE (абсолютный канон):\n{story_bible}\n\n" if story_bible else "")
            + f"КАНОНИЧЕСКИЕ ФАКТЫ ИЗ ПАМЯТИ:\n{canonical}\n\n"
            f"ПЛАН СЦЕНЫ:\n"
            f"Цель: {scene_plan.get('goal', '')}\n"
            f"Ключевое событие: {scene_plan.get('key_event', '')}\n\n"
            f"ТЕКУЩЕЕ СОСТОЯНИЕ:\n{memory_summary}\n\n"
            + (f"ПРЕДЫДУЩИЕ СЦЕНЫ:\n{previous_summary}\n\n" if previous_summary else "")
            + f"ТЕКСТ СЦЕНЫ ({word_count} слов):\n{scene_text}\n\n"
            f"Проверь на конфликты с Story Bible и каноническими фактами."
        )

        content, tokens = self._call_api(user_prompt)
        is_approved = content.strip().startswith(APPROVE_MARKER)

        if is_approved:
            self.total_approvals += 1
        else:
            self.total_rejections += 1

        return is_approved, content

    def get_stats(self) -> Dict[str, int]:
        return {
            "total_requests": self.total_requests,
            "total_approvals": self.total_approvals,
            "total_rejections": self.total_rejections,
        }

# ─────────────────────────── Вспомогательные функции ──────────────────────────
def _build_canonical_facts(memory_context: Dict[str, Any]) -> str:
    """
    Извлечь канонические факты которые НЕЛЬЗЯ нарушать.

    Это главный инструмент против путаницы в родстве и деталях.
    Включает: персонажей с ролями/родством, погоду, число подозреваемых.
    """
    lines: List[str] = []

    # ── Замороженные факты из памяти ──────────────────────────────────────────
    frozen_facts = memory_context.get("frozen_facts", {})
    frozen_present = frozen_facts.get("present", {})
    if frozen_present:
        lines.append("ЗАФИКСИРОВАННЫЕ ДЕТАЛИ МИРА:")
        for fact_key, fact_data in frozen_present.items():
            if isinstance(fact_data, dict):
                lines.append(f"  • {fact_key} = {fact_data.get('value', '?')}")

    # ── Персонажи из past (backstory — самый надёжный источник) ───────────────
    past = memory_context.get("past", {})
    past_chars = past.get("characters", {}) if isinstance(past, dict) else {}

    present = memory_context.get("present", {})
    present_chars = present.get("characters", {}) if isinstance(present, dict) else {}

    # Объединяем: past — канон, present — текущие данные
    all_chars: Dict[str, Any] = {}
    if isinstance(past_chars, dict):
        all_chars.update(past_chars)
    if isinstance(present_chars, dict):
        for name, info in present_chars.items():
            if name not in all_chars:
                all_chars[name] = info

    if all_chars:
        lines.append("\nПЕРСОНАЖИ (канонические роли и родство):")
        suspects = []
        investigator = None

        for name, info in all_chars.items():
            if not isinstance(info, dict):
                continue

            role = info.get("role", "")
            traits = info.get("traits", [])
            relationships = info.get("relationships", {})

            # Строим строку персонажа
            char_line = f"  • {name}"
            if role:
                char_line += f" — {role}"
            if traits:
                char_line += f" [{', '.join(traits[:3])}]"
            if relationships:
                rel_parts = [f"{k}: {v}" for k, v in relationships.items()]
                char_line += f" (родство/отношения: {'; '.join(rel_parts)})"

            lines.append(char_line)

            # Определяем подозреваемых vs следователя
            role_lower = role.lower()
            if any(w in role_lower for w in ["следователь", "инспектор", "детектив", "сыщик"]):
                investigator = name
            elif info.get("status") != "dead":
                suspects.append(name)

        # ── Явно указываем число подозреваемых ────────────────────────────────
        lines.append(f"\nЧИСЛО ПОДОЗРЕВАЕМЫХ: {len(suspects)}")
        lines.append(f"Подозреваемые: {', '.join(suspects)}")
        if investigator:
            lines.append(f"Следователь (НЕ подозреваемый): {investigator}")

    # ── Реестр улик ───────────────────────────────────────────────────────────
    clue_registry = memory_context.get("clue_registry", {})
    if isinstance(clue_registry, dict) and clue_registry:
        lines.append("\nЗАРЕГИСТРИРОВАННЫЕ УЛИКИ:")
        for clue_id, clue in clue_registry.items():
            if not isinstance(clue, dict):
                continue
            lines.append(
                f"  • [{clue_id}] {clue.get('description', '')} "
                f"— место: {clue.get('location', '?')}, "
                f"статус: {clue.get('status', 'hidden')}"
            )

    return "\n".join(lines) if lines else "Канонические факты не установлены"


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