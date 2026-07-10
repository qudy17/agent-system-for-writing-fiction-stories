"""
Агент-Писатель на базе DeepSeek V4 (через Yandex AI Studio).

Отвечает за:
    1. Генерацию начального лора и плана сцен (planning phase)
    2. Написание черновиков сцен на основе памяти
    3. Переписывание сцен по замечаниям Критика
    4. Генерацию diff-JSON с изменениями состояния мира после каждой сцены
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)


# ─────────────────────────── Системные промпты ────────────────────────────────

_SYSTEM_PLANNER = """Ты — мастер детективных историй. Твоя задача — создать
детальный лор и структуру детективного рассказа.

Ты ОБЯЗАН отвечать ТОЛЬКО валидным JSON без markdown-обёрток, без ```json,
без объяснений — только чистый JSON-объект.

Структура ответа:
{
  "title": "Название рассказа",
  "scene_plan": [
    {
      "index": 0,
      "title": "Название сцены",
      "goal": "Что должна раскрыть эта сцена",
      "key_event": "Ключевое событие сцены",
      "location": "Место действия"
    }
  ],
  "initial_lore": {
    "past": {
      "world_state": {"era": "...", "city": "...", "atmosphere": "..."},
      "characters": {
        "ИМЯ": {"role": "...", "status": "alive|dead", "location": "...", "traits": ["..."]}
      },
      "clues": [{"id": "...", "description": "...", "location": "...", "status": "hidden"}],
      "events": [{"description": "...", "timestamp": "...", "participants": ["..."]}],
      "secrets": ["..."]
    },
    "present": {
      "world_state": {"time_of_day": "...", "weather": "...", "mood": "..."},
      "characters": {},
      "clues": [],
      "events": [],
      "secrets": []
    },
    "future": {
      "world_state": {},
      "characters": {},
      "clues": [],
      "events": [],
      "secrets": ["финальное разоблачение: ..."]
    }
  }
}

Правила:
- Создай ровно 5 сцен
- Каждый персонаж должен иметь мотив
- Улики должны логично вести к разоблачению
- Разоблачение должно быть неожиданным, но логически обоснованным
- Возвращай ТОЛЬКО JSON, никакого другого текста
"""

_SYSTEM_WRITER = """Ты — талантливый автор детективных рассказов. Пишешь
захватывающие, атмосферные сцены в стиле нуар.

Твой ответ ВСЕГДА состоит из двух частей, разделённых маркером ---DIFF---:
1. Текст сцены (художественная проза)
2. JSON с изменениями памяти (diff)

Формат ответа:
<текст сцены — минимум 300 слов, художественная проза>
---DIFF---
{
  "past": {},
  "present": {
    "world_state": {"последние изменения": "..."},
    "characters": {"ИМЯ": {"location": "новое место", "status": "..."}},
    "clues": [{"id": "уникальный_id", "description": "...", "found_by": "...", "location": "...", "status": "found"}],
    "events": [{"description": "что произошло в этой сцене", "timestamp": "...", "participants": ["..."]}]
  },
  "future": {},
  "advance_scene_index": true
}

Правила написания:
- Пиши от третьего лица
- Используй сенсорные детали (звуки, запахи, ощущения)
- Каждая сцена должна заканчиваться крючком (hook), тянущим читателя вперёд
- Не раскрывай убийцу раньше финальной сцены
- Диалоги должны раскрывать характер персонажей
- ОБЯЗАТЕЛЬНО используй разделитель ---DIFF--- между текстом и JSON
"""

_SYSTEM_REWRITER = """Ты — опытный редактор детективных рассказов.
Тебе дают черновик сцены и список замечаний от Критика.
Твоя задача — исправить замечания, сохранив авторский стиль.

Ответ в том же формате — текст + ---DIFF--- + JSON.

Формат:
<исправленный текст сцены — минимум 300 слов>
---DIFF---
{
  "past": {},
  "present": {
    "world_state": {},
    "characters": {},
    "clues": [],
    "events": []
  },
  "future": {},
  "advance_scene_index": true
}

ОБЯЗАТЕЛЬНО используй разделитель ---DIFF--- между текстом и JSON.
"""


# ─────────────────────────── Агент-писатель ───────────────────────────────────

class WriterAgent:
    """
    Агент-Писатель на базе DeepSeek V4 через Yandex AI Studio.

    Использует OpenAI-compatible API эндпоинт Yandex Cloud.
    """

    def __init__(
        self,
        api_key: str,
        folder_id: str,
        model_name: str = "deepseek-v4-flash",
        base_url: str = "https://llm.api.cloud.yandex.net/v1",
        auth_scheme: str = "Api-Key",
        temperature: float = 0.7,
        max_tokens: int = 4000,
        timeout: int = 120,
        data_logging_enabled: bool = False,
    ):
        """
        Args:
            api_key               : Yandex Cloud API-ключ (или IAM-токен)
            folder_id             : Yandex Cloud folder ID
            model_name            : Имя модели (deepseek-v4-flash)
            base_url              : Базовый URL API
            auth_scheme           : "Api-Key" или "Bearer"
            temperature           : Температура генерации
            max_tokens            : Максимум токенов в ответе
            timeout               : Таймаут запроса в секундах
            data_logging_enabled  : Разрешить логирование данных Yandex
        """
        self.api_key = api_key
        self.folder_id = folder_id
        self.model_uri = f"gpt://{folder_id}/{model_name}"
        self.base_url = base_url.rstrip("/")
        self.auth_scheme = auth_scheme
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.data_logging_enabled = data_logging_enabled

        # Счётчики для логирования
        self.total_tokens_used: int = 0
        self.total_requests: int = 0

    def _build_headers(self) -> Dict[str, str]:
        """Собрать HTTP-заголовки для запроса к Yandex AI."""
        return {
            "Authorization": f"{self.auth_scheme} {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "x-folder-id": self.folder_id,
            "x-data-logging-enabled": str(self.data_logging_enabled).lower(),
        }

    def _call_api(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: Optional[float] = None,
    ) -> Tuple[str, int]:
        """
        Выполнить запрос к Yandex AI API (OpenAI-compatible).

        Args:
            system_prompt : Системная инструкция
            user_prompt   : Запрос пользователя
            temperature   : Переопределить температуру для конкретного вызова

        Returns:
            Tuple[ответ модели (str), использовано токенов (int)]

        Raises:
            requests.HTTPError : При ошибке HTTP
            RuntimeError       : При пустом / null контенте в ответе
        """
        start_time = time.time()
        temp = temperature if temperature is not None else self.temperature

        payload = {
            "model": self.model_uri,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temp,
            "max_tokens": self.max_tokens,
            "stream": False,
        }

        logger.debug(
            "Writer → запрос: model=%s, temp=%.2f, max_tokens=%d",
            self.model_uri, temp, self.max_tokens,
        )

        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=self._build_headers(),
            json=payload,
            timeout=self.timeout,
        )

        # ── Детальное логирование при HTTP-ошибке ─────────────────────────────
        if not response.ok:
            logger.error(
                "Writer: HTTP %d от Yandex AI\nURL: %s\nТело ответа: %s",
                response.status_code,
                response.url,
                response.text[:1000],
            )
            response.raise_for_status()

        data = response.json()

        # ── Логируем сырой ответ на уровне DEBUG для диагностики ──────────────
        logger.debug(
            "Writer ← сырой ответ: %s",
            json.dumps(data, ensure_ascii=False)[:800],
        )

        # ── Защита от отсутствующих полей в ответе ────────────────────────────
        choices = data.get("choices")
        if not choices:
            raise RuntimeError(
                f"Writer: поле 'choices' пустое или отсутствует.\n"
                f"Сырой ответ: {json.dumps(data, ensure_ascii=False)[:500]}"
            )

        message = choices[0].get("message")
        if message is None:
            raise RuntimeError(
                f"Writer: поле 'message' отсутствует в choices[0].\n"
                f"choices[0]: {json.dumps(choices[0], ensure_ascii=False)[:500]}"
            )

        # ── Проверяем finish_reason ────────────────────────────────────────────
        finish_reason = choices[0].get("finish_reason", "unknown")
        if finish_reason == "length":
            logger.warning(
                "⚠️  Writer: ответ обрезан по лимиту токенов (finish_reason=length). "
                "Увеличьте YANDEX_MAX_TOKENS в .env (сейчас: %d)",
                self.max_tokens,
            )

        content = message.get("content")

        # ── content может быть None при content_filter или сбое модели ────────
        if content is None:
            raise RuntimeError(
                f"Writer: content=null в ответе модели "
                f"(finish_reason={finish_reason!r}).\n"
                f"Возможные причины:\n"
                f"  1. finish_reason='content_filter' — контент заблокирован\n"
                f"  2. finish_reason='length' — обрезано по токенам\n"
                f"  3. Временный сбой API\n"
                f"Полный ответ: {json.dumps(data, ensure_ascii=False)[:800]}"
            )

        content = content.strip()
        if not content:
            raise RuntimeError(
                f"Writer: content пустой после strip() "
                f"(finish_reason={finish_reason!r})"
            )

        tokens = data.get("usage", {}).get("total_tokens", 0)
        elapsed = time.time() - start_time

        self.total_tokens_used += tokens
        self.total_requests += 1

        logger.debug(
            "✍️  Writer API: %.1f сек, %d токенов (всего: %d)",
            elapsed, tokens, self.total_tokens_used,
        )

        return content, tokens

    # ── Публичные методы ───────────────────────────────────────────────────────

    def plan_story(self, initial_prompt: str) -> Dict[str, Any]:
        """
        ФАЗА ПЛАНИРОВАНИЯ: Сгенерировать лор и план сцен.

        Args:
            initial_prompt : Стартовая идея/сеттинг для истории

        Returns:
            {title, scene_plan, initial_lore} — распарсенный JSON
        """
        logger.info("📖 Writer: Планирую историю...")

        user_prompt = (
            f"Создай детективный рассказ на основе следующей идеи:\n\n"
            f"{initial_prompt}\n\n"
            f"Обязательно включи неожиданного убийцу с логически обоснованным мотивом. "
            f"История должна быть из 5 сцен. "
            f"Верни ТОЛЬКО валидный JSON без каких-либо пояснений и markdown."
        )

        content, tokens = self._call_api(
            system_prompt=_SYSTEM_PLANNER,
            user_prompt=user_prompt,
            temperature=0.8,
        )

        logger.info("✍️  Writer: Получен план (%d токенов)", tokens)
        return _extract_json_safe(content)

    def write_scene(
        self,
        scene_plan: Dict[str, Any],
        memory_context: Dict[str, Any],
        approved_scenes: List[Dict[str, Any]],
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Написать черновик сцены на основе плана и памяти.

        Args:
            scene_plan      : {index, title, goal, key_event, location}
            memory_context  : Текущее состояние из MCP (all layers)
            approved_scenes : Уже написанные сцены для контекста

        Returns:
            Tuple[текст сцены, diff для памяти]
        """
        logger.info(
            "✍️  Writer: Пишу сцену %d — '%s'...",
            scene_plan.get("index", "?"),
            scene_plan.get("title", ""),
        )

        # Берём только последние 2 сцены, чтобы не перегружать контекст
        previous_text = ""
        if approved_scenes:
            last_scenes = approved_scenes[-2:]
            previous_text = "\n\n".join(
                f"=== {s['title']} ===\n{s['text']}" for s in last_scenes
            )

        # Компактная версия памяти — только текущий слой
        memory_summary = _compact_memory(memory_context)

        user_prompt = (
            f"ТЕКУЩАЯ СЦЕНА ДЛЯ НАПИСАНИЯ:\n"
            f"Номер: {scene_plan.get('index', 0) + 1}\n"
            f"Название: {scene_plan.get('title', '')}\n"
            f"Цель сцены: {scene_plan.get('goal', '')}\n"
            f"Ключевое событие: {scene_plan.get('key_event', '')}\n"
            f"Место: {scene_plan.get('location', '')}\n\n"
            f"ТЕКУЩЕЕ СОСТОЯНИЕ МИРА (память):\n"
            f"{memory_summary}\n\n"
            + (
                f"ПРЕДЫДУЩИЕ СЦЕНЫ (для связности):\n{previous_text}\n\n"
                if previous_text else ""
            )
            + "Напиши сцену. ОБЯЗАТЕЛЬНО раздели текст и JSON маркером ---DIFF---"
        )

        content, tokens = self._call_api(
            system_prompt=_SYSTEM_WRITER,
            user_prompt=user_prompt,
        )

        logger.info("✍️  Writer: Черновик написан (%d токенов)", tokens)
        return _parse_writer_response(content)

    def rewrite_scene(
        self,
        original_text: str,
        critic_feedback: str,
        scene_plan: Dict[str, Any],
        memory_context: Dict[str, Any],
        iteration: int,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Переписать сцену по замечаниям Критика.

        Args:
            original_text   : Исходный черновик
            critic_feedback : Замечания от GigaChat
            scene_plan      : План текущей сцены
            memory_context  : Текущая память
            iteration       : Номер итерации (для логирования)

        Returns:
            Tuple[исправленный текст, новый diff]
        """
        logger.info(
            "✍️  Writer: Переписываю сцену '%s' (итерация %d)...",
            scene_plan.get("title", ""),
            iteration,
        )

        memory_summary = _compact_memory(memory_context)

        user_prompt = (
            f"ЧЕРНОВИК СЦЕНЫ:\n{original_text}\n\n"
            f"ЗАМЕЧАНИЯ КРИТИКА:\n{critic_feedback}\n\n"
            f"ПЛАН СЦЕНЫ:\n"
            f"Цель: {scene_plan.get('goal', '')}\n"
            f"Ключевое событие: {scene_plan.get('key_event', '')}\n\n"
            f"ТЕКУЩАЯ ПАМЯТЬ:\n{memory_summary}\n\n"
            f"Исправь все замечания Критика, сохрани авторский стиль. "
            f"ОБЯЗАТЕЛЬНО раздели текст и JSON маркером ---DIFF---"
        )

        content, tokens = self._call_api(
            system_prompt=_SYSTEM_REWRITER,
            user_prompt=user_prompt,
            temperature=0.5,
        )

        logger.info(
            "✍️  Writer: Переписана (итерация %d, %d токенов)", iteration, tokens
        )
        return _parse_writer_response(content)

    def get_stats(self) -> Dict[str, int]:
        """Вернуть статистику использования."""
        return {
            "total_requests": self.total_requests,
            "total_tokens": self.total_tokens_used,
        }


# ─────────────────────────── Вспомогательные функции ──────────────────────────

def _compact_memory(memory_context: Dict[str, Any]) -> str:
    """
    Создать компактное текстовое представление памяти.

    Используем текстовый формат вместо полного JSON,
    чтобы экономить токены и не перегружать контекст модели.
    """
    lines: List[str] = []

    for layer in ["past", "present", "future"]:
        layer_data = memory_context.get(layer)
        if not layer_data or not isinstance(layer_data, dict):
            continue

        has_content = any(
            layer_data.get(k)
            for k in ["world_state", "characters", "clues", "events", "secrets"]
        )
        if not has_content:
            continue

        lines.append(f"[{layer.upper()}]")

        world = layer_data.get("world_state") or {}
        if world:
            world_str = ", ".join(f"{k}: {v}" for k, v in world.items() if v)
            if world_str:
                lines.append(f"  Мир: {world_str}")

        chars = layer_data.get("characters") or {}
        if chars:
            for name, info in chars.items():
                if isinstance(info, dict):
                    loc = info.get("location", "неизвестно")
                    status = info.get("status", "")
                    role = info.get("role", "")
                    traits = ", ".join(info.get("traits") or [])
                    char_line = f"  {name} ({role}): статус={status}, место={loc}"
                    if traits:
                        char_line += f", черты={traits}"
                    lines.append(char_line)

        clues = layer_data.get("clues") or []
        if clues:
            lines.append("  Улики:")
            for clue in clues:
                if isinstance(clue, dict):
                    lines.append(
                        f"    - [{clue.get('id', '?')}] "
                        f"{clue.get('description', '')} "
                        f"(статус: {clue.get('status', 'unknown')})"
                    )

        events = layer_data.get("events") or []
        if events:
            lines.append("  События:")
            for event in events[-3:]:
                if isinstance(event, dict):
                    lines.append(f"    - {event.get('description', '')}")

        secrets = layer_data.get("secrets") or []
        if secrets:
            lines.append("  Тайны:")
            for secret in secrets:
                lines.append(f"    - {secret}")

    return "\n".join(lines) if lines else "Память пуста"


def _extract_json_safe(text: str) -> Dict[str, Any]:
    """
    Безопасно извлечь JSON из текста модели.

    Пробует несколько стратегий:
        1. Прямой парсинг (если текст уже чистый JSON)
        2. Удаление markdown-обёрток (```json ... ```)
        3. Поиск JSON-блока регулярным выражением
        4. Попытка исправить обрезанный JSON

    Raises:
        ValueError: Если ни одна стратегия не сработала
    """
    # Стратегия 1: Прямой парсинг
    cleaned = text.strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Стратегия 2: Убираем markdown-обёртки
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"\s*```\s*$", "", cleaned, flags=re.MULTILINE)
    cleaned = cleaned.strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Стратегия 3: Ищем JSON-блок через регулярное выражение
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    # Стратегия 4: Пробуем исправить обрезанный JSON (добавляем закрывающие скобки)
    if "{" in cleaned:
        candidate = match.group(0) if match else cleaned
        open_braces = candidate.count("{") - candidate.count("}")
        open_brackets = candidate.count("[") - candidate.count("]")
        fixed = candidate + ("]" * max(0, open_brackets)) + ("}" * max(0, open_braces))
        try:
            return json.loads(fixed)
        except json.JSONDecodeError:
            pass

    logger.error("Writer: не удалось распарсить JSON из ответа:\n%s", text[:800])
    raise ValueError(
        f"Не удалось извлечь JSON из ответа Writer.\n"
        f"Первые 300 символов ответа: {text[:300]}"
    )


def _parse_writer_response(content: str) -> Tuple[str, Dict[str, Any]]:
    """
    Разобрать ответ Писателя на текст сцены и diff-JSON.

    Ожидаемый формат:
        <текст сцены>
        ---DIFF---
        {json diff}

    Если маркер отсутствует — весь контент считается текстом сцены,
    diff заполняется значениями по умолчанию.

    Returns:
        Tuple[текст сцены, diff словарь]

    Raises:
        ValueError: Если текст сцены пустой
    """
    separator = "---DIFF---"

    if separator in content:
        parts = content.split(separator, 1)
        scene_text = parts[0].strip()
        diff_raw = parts[1].strip()

        try:
            diff = _extract_json_safe(diff_raw)
        except ValueError:
            logger.warning(
                "Writer: не удалось распарсить DIFF после маркера, "
                "использую пустой diff"
            )
            diff = _make_empty_diff()
    else:
        # Модель не соблюла формат — проверяем, нет ли JSON в конце текста
        logger.warning(
            "Writer: ответ без маркера ---DIFF---, пытаюсь найти JSON в конце"
        )

        # Ищем последний JSON-блок в тексте
        json_matches = list(re.finditer(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", content, re.DOTALL))
        if json_matches:
            last_match = json_matches[-1]
            potential_diff_raw = last_match.group(0)
            try:
                diff = _extract_json_safe(potential_diff_raw)
                # Убираем найденный JSON из текста сцены
                scene_text = content[: last_match.start()].strip()
                logger.info("Writer: JSON найден в конце текста без маркера")
            except ValueError:
                scene_text = content.strip()
                diff = _make_empty_diff()
        else:
            scene_text = content.strip()
            diff = _make_empty_diff()

    if not scene_text:
        raise ValueError("Writer: пустой текст сцены после парсинга ответа")

    # Гарантируем наличие advance_scene_index в diff
    if "advance_scene_index" not in diff:
        diff["advance_scene_index"] = True

    return scene_text, diff


def _make_empty_diff() -> Dict[str, Any]:
    """Вернуть безопасный пустой diff по умолчанию."""
    return {
        "past": {},
        "present": {
            "events": [],
            "characters": {},
            "clues": [],
            "world_state": {},
        },
        "future": {},
        "advance_scene_index": True,
    }