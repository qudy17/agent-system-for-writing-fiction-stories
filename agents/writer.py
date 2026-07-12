"""
Агент-Писатель на базе DeepSeek V4 (через Yandex AI Studio).
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)

SCENE_MIN_WORDS = 300
SCENE_MAX_WORDS = 550
SCENE_HARD_LIMIT = 600


def _count_words(text: str) -> int:
    """Подсчитать количество слов в тексте."""
    return len(text.split())


def _truncate_to_word_limit(text: str, max_words: int) -> str:
    """
    Обрезать текст до max_words слов, заканчивая на границе предложения.
    """
    words = text.split()
    if len(words) <= max_words:
        return text

    truncated = " ".join(words[:max_words])

    for punct in (".", "!", "?", "…"):
        last_punct = truncated.rfind(punct)
        if last_punct > len(truncated) * 0.7:
            return truncated[: last_punct + 1]

    return truncated + "..."


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
      "goal": "Что должна раскрыть эта сцена (1 предложение)",
      "key_event": "Одно ключевое событие сцены",
      "location": "Место действия"
    }
  ],
  "initial_lore": {
    "past": {
      "world_state": {"era": "...", "city": "...", "atmosphere": "..."},
      "characters": {
        "ИМЯ": {
          "role": "...",
          "status": "alive|dead",
          "location": "...",
          "traits": ["..."],
          "relationships": {"ДРУГОЙ_ПЕРСОНАЖ": "тип отношений"}
        }
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
- Каждая сцена — ОДНО ключевое событие, не несколько
- Каждый персонаж должен иметь мотив
- У каждого персонажа обязательно заполни поле relationships
- Улики должны логично вести к разоблачению
- Разоблачение должно быть неожиданным, но логически обоснованным
- Возвращай ТОЛЬКО JSON, никакого другого текста
"""

_SYSTEM_WRITER = """Ты — талантливый автор детективных рассказов. Пишешь
захватывающие, атмосферные сцены в стиле нуар.

ВАЖНЕЙШЕЕ ТРЕБОВАНИЕ К ДЛИНЕ:
- Сцена должна быть строго 300–550 слов (НЕ БОЛЬШЕ 550 СЛОВ)
- Считай слова по мере написания и останавливайся на 500-550
- Это жёсткое ограничение: сцена свыше 600 слов будет отклонена

ЗАПРЕТ НА ПОВТОРЕНИЯ (критически важно!):
- Если персонаж уже осматривал улику в предыдущей сцене — не описывай
  осмотр заново. Ссылайся кратко: "волос, замеченный ранее"
- Если локация уже была описана — не воспроизводи её атмосферу слово в слово.
  Один новый сенсорный штрих вместо полного переописания
- Каждая деталь (запах, предмет, улика) должна упоминаться в ОДНОМ каноническом
  месте. В следующих сценах — только ссылка, не переописание
- Проверяй: если фраза из предыдущей сцены почти дословно совпадает
  с тем что ты пишешь — ОСТАНОВИ себя и найди другой способ выразить мысль

ПРАВИЛО ОДНОГО ПИСЬМА:
- Если документ (письмо, записка) уже был процитирован в предыдущей сцене —
  его текст ЗАМОРОЖЕН. Нельзя дописывать, переформулировать или цитировать
  другие слова из него. Только ссылка: "то самое письмо" или "незаконченное
  письмо Зимберга"

ПРАВИЛО ОДНОГО МЕСТА:
- Каждый предмет находится ровно в одном месте. Если сапфир лежал на столике
  в сцене 3 — в сцене 5 он не может "оказаться за зеркалом" без явного
  описания того КАК и КОГДА он был перенесён

Твой ответ ВСЕГДА состоит из двух частей, разделённых маркером ---DIFF---:
1. Текст сцены (художественная проза, 300–550 слов)
2. JSON с изменениями памяти (diff)

Формат ответа:
<текст сцены — строго 300–550 слов, художественная проза>
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
- ОДНО ключевое событие на сцену — не распыляйся
- Используй сенсорные детали (звуки, запахи, ощущения)
- Каждая сцена должна заканчиваться крючком (hook)
- Не раскрывай убийцу раньше финальной сцены
- Диалоги — максимум 3–4 реплики на сцену
- СТРОГО следуй Story Bible если она предоставлена
- ОБЯЗАТЕЛЬНО используй разделитель ---DIFF--- между текстом и JSON
"""

_SYSTEM_REWRITER = """Ты — опытный редактор детективных рассказов.
Тебе дают черновик сцены и список замечаний от Критика.
Твоя задача — исправить замечания, сохранив авторский стиль.

ВАЖНЕЙШЕЕ ТРЕБОВАНИЕ К ДЛИНЕ:
- Сцена должна быть строго 300–550 слов (НЕ БОЛЬШЕ 550 СЛОВ)
- Если черновик длиннее — сократи, убрав лишние описания и диалоги
- Оставь только самое важное для этой сцены

СТРОГО следуй Story Bible если она предоставлена:
- Имена персонажей нельзя менять
- Родство и роли нельзя менять
- Местонахождение предметов должно соответствовать Bible

Ответ в том же формате — текст + ---DIFF--- + JSON.

Формат:
<исправленный текст сцены — строго 300–550 слов>
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
    """

    def __init__(
        self,
        api_key: str,
        folder_id: str,
        model_name: str = "deepseek-v4-flash",
        base_url: str = "https://llm.api.cloud.yandex.net/v1",
        auth_scheme: str = "Api-Key",
        temperature: float = 0.7,
        max_tokens: int = 10000,
        timeout: int = 120,
        data_logging_enabled: bool = False,
    ):
        self.api_key = api_key
        self.folder_id = folder_id
        self.model_uri = f"gpt://{folder_id}/{model_name}"
        self.base_url = base_url.rstrip("/")
        self.auth_scheme = auth_scheme
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.data_logging_enabled = data_logging_enabled

        self.total_tokens_used: int = 0
        self.total_requests: int = 0

    def _build_headers(self) -> Dict[str, str]:
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

        Returns:
            Tuple[ответ модели (str), использовано токенов (int)]
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

        if not response.ok:
            logger.error(
                "Writer: HTTP %d от Yandex AI\nURL: %s\nТело: %s",
                response.status_code,
                response.url,
                response.text[:1000],
            )
            response.raise_for_status()

        data = response.json()

        logger.debug(
            "Writer ← сырой ответ: %s",
            json.dumps(data, ensure_ascii=False)[:800],
        )

        choices = data.get("choices")
        if not choices:
            raise RuntimeError(
                f"Writer: поле 'choices' пустое.\n"
                f"Ответ: {json.dumps(data, ensure_ascii=False)[:500]}"
            )

        message = choices[0].get("message")
        if message is None:
            raise RuntimeError(
                f"Writer: поле 'message' отсутствует.\n"
                f"choices[0]: {json.dumps(choices[0], ensure_ascii=False)[:500]}"
            )

        finish_reason = choices[0].get("finish_reason", "unknown")
        if finish_reason == "length":
            logger.warning(
                "⚠️  Writer: ответ обрезан (finish_reason=length). "
                "Увеличьте YANDEX_MAX_TOKENS (сейчас: %d)",
                self.max_tokens,
            )

        content = message.get("content")

        if content is None:
            raise RuntimeError(
                f"Writer: content=null (finish_reason={finish_reason!r}).\n"
                f"Ответ: {json.dumps(data, ensure_ascii=False)[:800]}"
            )

        content = content.strip()
        if not content:
            raise RuntimeError(
                f"Writer: content пустой (finish_reason={finish_reason!r})"
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
            {title, scene_plan, initial_lore}
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
        story_bible: str = "",
        approved_plan: str = "",
        scene_transition: str = "",
    ) -> Tuple[str, Dict[str, Any]]:

        previous_text = ""
        if approved_scenes:
            last_scene = approved_scenes[-1]
            previous_text = (
                f"=== {last_scene['title']} ===\n{last_scene['text']}"
            )

        # ── НОВОЕ: Строим список уже использованных деталей ───────────────────
        used_details = _build_used_details_warning(approved_scenes)

        memory_summary = _compact_memory(memory_context)

        user_prompt = (
            (f"{story_bible}\n\n" if story_bible else "")
            + (f"{scene_transition}\n\n" if scene_transition else "")
            + (
                f"✅ ОДОБРЕННЫЙ ПЛАН СЦЕНЫ (строго следуй ему!):\n"
                f"{approved_plan}\n\n"
                if approved_plan else ""
            )
            # ── НОВОЕ: предупреждение о повторениях ───────────────────────────
            + (f"{used_details}\n\n" if used_details else "")
            + f"СЦЕНА ДЛЯ НАПИСАНИЯ:\n"
            f"Номер: {scene_plan.get('index', 0) + 1} из 5\n"
            f"Название: {scene_plan.get('title', '')}\n"
            f"Цель: {scene_plan.get('goal', '')}\n"
            f"Ключевое событие: {scene_plan.get('key_event', '')}\n"
            f"Место: {scene_plan.get('location', '')}\n\n"
            f"⚠️  ЛИМИТ: 300–550 слов. Одно ключевое событие.\n"
            f"⚠️  ЗАПРЕЩЕНО: упоминать события «за кадром» или «час назад».\n"
            f"⚠️  ЗАПРЕЩЕНО: вводить новые предметы (сейфы, украшения)\n"
            f"    которых не было в предыдущих сценах.\n\n"
            f"СОСТОЯНИЕ МИРА:\n{memory_summary}\n\n"
            + (
                f"ПРЕДЫДУЩАЯ СЦЕНА (НЕ ПОВТОРЯЙ её детали дословно!):\n"
                f"{previous_text}\n\n"
                if previous_text else ""
            )
            + "Напиши художественную сцену по одобренному плану. "
            "---DIFF--- обязателен."
        )

        content, tokens = self._call_api(
            system_prompt=_SYSTEM_WRITER,
            user_prompt=user_prompt,
        )

        scene_text, diff = _parse_writer_response(content)
        scene_text = _validate_and_fix_length(
            scene_text,
            scene_index=scene_plan.get("index", 0),
            context="write_scene",
        )

        logger.info("✍️  Writer: Черновик написан (%d токенов)", tokens)
        return scene_text, diff
    
    def rewrite_scene(
        self,
        original_text: str,
        critic_feedback: str,
        scene_plan: Dict[str, Any],
        memory_context: Dict[str, Any],
        iteration: int,
        story_bible: str = "",                          # ← Story Bible
        conflicts: Optional[List[Dict[str, Any]]] = None,  # ← конфликты из StateTracker
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Переписать сцену по замечаниям Критика.

        Args:
            original_text   : Исходный черновик
            critic_feedback : Замечания от GigaChat
            scene_plan      : План текущей сцены
            memory_context  : Текущая память
            iteration       : Номер итерации
            story_bible     : Канонический текст Story Bible
            conflicts       : Критические конфликты из StateTracker

        Returns:
            Tuple[исправленный текст, новый diff]
        """
        logger.info(
            "✍️  Writer: Переписываю сцену '%s' (итерация %d)...",
            scene_plan.get("title", ""),
            iteration,
        )

        memory_summary = _compact_memory(memory_context)
        word_count = _count_words(original_text)

        # Форматируем критические конфликты отдельным блоком
        conflicts_text = ""
        if conflicts:
            critical = [c for c in conflicts if c.get("severity") == "critical"]
            if critical:
                conflicts_text = "\n⛔ КРИТИЧЕСКИЕ КОНФЛИКТЫ (ИСПРАВЬ ОБЯЗАТЕЛЬНО):\n"
                for c in critical:
                    conflicts_text += (
                        f"  • {c.get('description', '')}\n"
                        f"    Канон: '{c.get('canonical_fact', '')}'\n"
                        f"    В тексте: '{c.get('scene_fact', '')}'\n"
                    )

        user_prompt = (
            (f"{story_bible}\n\n" if story_bible else "")
            + f"ЧЕРНОВИК СЦЕНЫ ({word_count} слов):\n{original_text}\n\n"
            f"ЗАМЕЧАНИЯ КРИТИКА:\n{critic_feedback}\n"
            + conflicts_text
            + f"\nПЛАН СЦЕНЫ:\n"
            f"Цель: {scene_plan.get('goal', '')}\n"
            f"Ключевое событие: {scene_plan.get('key_event', '')}\n\n"
            f"⚠️  ЛИМИТ: 300–550 слов. "
            + ("Текущий черновик превышает лимит — сократи!\n\n"
               if word_count > SCENE_MAX_WORDS else "\n\n")
            + f"СОСТОЯНИЕ МИРА:\n{memory_summary}\n\n"
            f"Исправь ВСЕ замечания. Story Bible — абсолютный приоритет. "
            f"---DIFF--- обязателен."
        )

        content, tokens = self._call_api(
            system_prompt=_SYSTEM_REWRITER,
            user_prompt=user_prompt,
            temperature=0.5,
        )

        scene_text, diff = _parse_writer_response(content)

        scene_text = _validate_and_fix_length(
            scene_text,
            scene_index=scene_plan.get("index", 0),
            context=f"rewrite_scene_iter{iteration}",
        )

        logger.info(
            "✍️  Writer: Переписана (итерация %d, %d токенов)", iteration, tokens
        )
        return scene_text, diff

    def get_stats(self) -> Dict[str, int]:
        """Вернуть статистику использования."""
        return {
            "total_requests": self.total_requests,
            "total_tokens": self.total_tokens_used,
        }


# ─────────────────────────── Вспомогательные функции ──────────────────────────

def _validate_and_fix_length(
    scene_text: str,
    scene_index: int,
    context: str,
) -> str:
    """
    Проверить длину сцены и при необходимости обрезать.

    Args:
        scene_text  : Текст сцены
        scene_index : Индекс сцены (для логирования)
        context     : Контекст вызова (для логирования)

    Returns:
        Текст сцены, не превышающий SCENE_HARD_LIMIT слов.
    """
    word_count = _count_words(scene_text)

    logger.info(
        "📏 Сцена %d [%s]: %d слов (лимит: %d–%d)",
        scene_index + 1, context, word_count,
        SCENE_MIN_WORDS, SCENE_MAX_WORDS,
    )

    if word_count < SCENE_MIN_WORDS:
        logger.warning(
            "⚠️  Сцена %d слишком короткая: %d слов (минимум: %d)",
            scene_index + 1, word_count, SCENE_MIN_WORDS,
        )
        # Короткие сцены не обрезаем — критик попросит дополнить

    elif word_count > SCENE_HARD_LIMIT:
        logger.warning(
            "✂️  Сцена %d слишком длинная (%d слов), обрезаю до %d...",
            scene_index + 1, word_count, SCENE_MAX_WORDS,
        )
        scene_text = _truncate_to_word_limit(scene_text, SCENE_MAX_WORDS)
        new_count = _count_words(scene_text)
        logger.info("✂️  После обрезки: %d слов", new_count)

    return scene_text


def _compact_memory(memory_context: Dict[str, Any]) -> str:
    """
    Создать компактное текстовое представление памяти.

    Используется как дополнение к Story Bible —
    содержит текущее состояние мира (локации персонажей, последние события).
    """
    lines: List[str] = []

    # ── Замороженные факты ─────────────────────────────────────────────────────
    frozen_facts = memory_context.get("frozen_facts")
    if isinstance(frozen_facts, dict):
        frozen_present = frozen_facts.get("present")
        if isinstance(frozen_present, dict) and frozen_present:
            lines.append("⚠️  ЗАФИКСИРОВАННЫЕ ФАКТЫ (не менять!):")
            for fact_key, fact_data in frozen_present.items():
                if isinstance(fact_data, dict):
                    lines.append(
                        f"  🔒 {fact_key} = {fact_data.get('value', '?')}"
                    )
            lines.append("")

    # ── Реестр улик ────────────────────────────────────────────────────────────
    clue_registry = memory_context.get("clue_registry")
    if isinstance(clue_registry, dict) and clue_registry:
        lines.append("📋 ВСЕ УЛИКИ В ИСТОРИИ:")
        for clue_id, clue in clue_registry.items():
            if not isinstance(clue, dict):
                continue
            status_icon = "🔍" if clue.get("status") == "found" else "🔒"
            lines.append(
                f"  {status_icon} [{clue_id}] {clue.get('description', '')} "
                f"— {clue.get('location', '?')} "
                f"(сцена {clue.get('introduced_in_scene', 0) + 1}, "
                f"статус: {clue.get('status', 'hidden')})"
            )
        lines.append("")

    # ── Временные слои ─────────────────────────────────────────────────────────
    for layer in ["past", "present", "future"]:
        layer_data = memory_context.get(layer)
        if not isinstance(layer_data, dict):
            continue

        has_content = any(
            layer_data.get(k)
            for k in ["world_state", "characters", "events", "secrets"]
        )
        if not has_content:
            continue

        lines.append(f"[{layer.upper()}]")

        # world_state
        world = layer_data.get("world_state")
        if isinstance(world, dict):
            world_parts = [
                f"{k}: {v}"
                for k, v in world.items()
                if v is not None and k != "frozen_facts"
            ]
            if world_parts:
                lines.append(f"  Мир: {', '.join(world_parts)}")
        elif isinstance(world, str) and world:
            lines.append(f"  Мир: {world}")

        # Персонажи
        chars = layer_data.get("characters")
        if isinstance(chars, dict):
            for name, info in chars.items():
                if not isinstance(info, dict):
                    lines.append(f"  {name}: {info}")
                    continue
                parts = [f"  {name}"]
                if info.get("role"):
                    parts.append(f"({info['role']})")
                parts.append(f"статус={info.get('status', '?')}")
                parts.append(f"место={info.get('location', '?')}")
                lines.append(" ".join(parts))

        # События (последние 3)
        events = layer_data.get("events")
        if isinstance(events, list) and events:
            lines.append("  События:")
            for event in events[-3:]:
                if isinstance(event, dict):
                    lines.append(f"    - {event.get('description', '')}")
                elif isinstance(event, str):
                    lines.append(f"    - {event}")

        # Тайны
        secrets = layer_data.get("secrets")
        if isinstance(secrets, list) and secrets:
            lines.append("  Тайны:")
            for secret in secrets:
                lines.append(f"    - {secret}")

    return "\n".join(lines) if lines else "Память пуста"


def _extract_json_safe(text: str) -> Dict[str, Any]:
    """
    Безопасно извлечь JSON из текста модели.

    Стратегии (в порядке применения):
        1. Прямой парсинг
        2. Удаление markdown-обёрток
        3. Поиск JSON-блока регулярным выражением
        4. Попытка исправить обрезанный JSON

    Raises:
        ValueError: Если ни одна стратегия не сработала.
    """
    cleaned = text.strip()

    # Стратегия 1
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Стратегия 2
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"\s*```\s*$", "", cleaned, flags=re.MULTILINE)
    cleaned = cleaned.strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Стратегия 3
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    # Стратегия 4: исправляем обрезанный JSON
    if "{" in cleaned:
        candidate = match.group(0) if match else cleaned
        open_braces = candidate.count("{") - candidate.count("}")
        open_brackets = candidate.count("[") - candidate.count("]")
        fixed = (
            candidate
            + ("]" * max(0, open_brackets))
            + ("}" * max(0, open_braces))
        )
        try:
            return json.loads(fixed)
        except json.JSONDecodeError:
            pass

    logger.error("Writer: не удалось распарсить JSON:\n%s", text[:800])
    raise ValueError(
        f"Не удалось извлечь JSON из ответа Writer.\n"
        f"Первые 300 символов: {text[:300]}"
    )


def _parse_writer_response(content: str) -> Tuple[str, Dict[str, Any]]:
    """
    Разобрать ответ Писателя на текст сцены и diff-JSON.

    Ожидаемый формат:
        <текст сцены>
        ---DIFF---
        {json diff}

    Если маркер отсутствует — ищем JSON в конце текста.
    Если JSON не найден — возвращаем пустой diff.

    Returns:
        Tuple[текст сцены, diff словарь]

    Raises:
        ValueError: Если текст сцены пустой.
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
                "Writer: не удалось распарсить DIFF, использую пустой diff"
            )
            diff = _make_empty_diff()
    else:
        logger.warning(
            "Writer: ответ без ---DIFF---, ищу JSON в конце текста"
        )
        json_matches = list(
            re.finditer(
                r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", content, re.DOTALL
            )
        )
        if json_matches:
            last_match = json_matches[-1]
            try:
                diff = _extract_json_safe(last_match.group(0))
                scene_text = content[: last_match.start()].strip()
                logger.info("Writer: JSON найден в конце без маркера")
            except ValueError:
                scene_text = content.strip()
                diff = _make_empty_diff()
        else:
            scene_text = content.strip()
            diff = _make_empty_diff()

    if not scene_text:
        raise ValueError("Writer: пустой текст сцены после парсинга")

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

def _build_used_details_warning(
    approved_scenes: List[Dict[str, Any]],
) -> str:
    """
    Собрать список деталей из предыдущих сцен которые нельзя повторять.

    Извлекает первые предложения каждой сцены как "fingerprint" —
    если Writer видит похожий текст в своём черновике, это сигнал повторения.

    Args:
        approved_scenes : Уже одобренные сцены

    Returns:
        Предупреждающий блок для промпта или пустая строка.
    """
    if not approved_scenes:
        return ""

    lines = [
        "⛔ УЖЕ ОПИСАННЫЕ ДЕТАЛИ — НЕ ПОВТОРЯЙ ДОСЛОВНО:",
    ]

    for scene in approved_scenes:
        title = scene.get("title", "?")
        text = scene.get("text", "")
        if not text:
            continue

        # Берём первые 2 предложения как fingerprint сцены
        sentences = [s.strip() for s in text.replace("!", ".").replace("?", ".").split(".") if s.strip()]
        fingerprint = ". ".join(sentences[:2]) if sentences else text[:100]

        lines.append(f"  [{title}]: «{fingerprint[:120]}...»")

    lines.append(
        "  Если твой текст начинается похожими словами — "
        "перепиши с другого угла зрения."
    )

    return "\n".join(lines)