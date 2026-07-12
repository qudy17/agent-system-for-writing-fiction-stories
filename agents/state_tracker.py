"""
Global State Tracker — агент извлечения и валидации фактов.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)


# ─────────────────────────── Вспомогательные функции ──────────────────────────
# Определяем ДО класса, чтобы и класс и внешний код могли их использовать

def fix_truncated_json(text: str) -> str:
    """
    Починить обрезанный JSON добавив закрывающие скобки.

    Алгоритм:
        1. Убираем незавершённую последнюю строку
        2. Закрываем незакрытые строковые литералы
        3. Закрываем незакрытые массивы []
        4. Закрываем незакрытые объекты {}

    Args:
        text : Обрезанный JSON-текст начинающийся с {

    Returns:
        Починенный JSON-текст
    """
    if not text:
        return text

    # Убираем последнюю строку если она незавершена
    lines = text.rstrip().split("\n")
    last_line = lines[-1].rstrip() if lines else ""

    if last_line and not any(
        last_line.endswith(c) for c in (
            "}", "]", '",', '"', ",",
            "true,", "false,", "true", "false",
        )
    ):
        # Нечётное число кавычек → строка не закрыта → убираем строку
        if last_line.count('"') % 2 != 0:
            lines = lines[:-1]
            text = "\n".join(lines)

    # Считаем незакрытые скобки вне строк
    in_string = False
    escape_next = False
    depth_curly = 0
    depth_square = 0

    for char in text:
        if escape_next:
            escape_next = False
            continue
        if char == "\\" and in_string:
            escape_next = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            depth_curly += 1
        elif char == "}":
            depth_curly -= 1
        elif char == "[":
            depth_square += 1
        elif char == "]":
            depth_square -= 1

    # Закрываем незакрытую строку
    if in_string:
        text += '"'

    # Закрываем массивы и объекты (порядок важен: массивы глубже)
    text += "]" * max(0, depth_square)
    text += "}" * max(0, depth_curly)

    return text


# ─────────────────────────── Промпты ──────────────────────────────────────────

_SYSTEM_EXTRACTOR = """Ты — аналитик текста. Извлекаешь факты из сцены детективного рассказа.

Возвращай ТОЛЬКО валидный JSON без markdown, без объяснений.
JSON должен быть компактным — краткие значения (до 10 слов каждое).

{
  "characters_mentioned": [
    {
      "name": "Имя",
      "role": "роль",
      "status": "alive|dead|unknown",
      "location": "место (кратко)",
      "actions": ["действие1", "действие2"]
    }
  ],
  "items_mentioned": [
    {
      "name": "предмет",
      "status": "на месте|украден|найден|уничтожен|у персонажа",
      "location": "где (кратко)",
      "holder": "у кого или null"
    }
  ],
  "documents_mentioned": [
    {
      "name": "название документа",
      "exact_text": "точная цитата текста если приведена в сцене",
      "status": "finished|unfinished|sealed",
      "location": "где находится"
    }
  ],
  "doors_mentioned": [
    {
      "name": "название двери/прохода",
      "locked_status": "locked_from_outside|locked_from_inside|unlocked|open",
      "key_location": "где ключ (кратко)",
      "key_holder": "у кого ключ"
    }
  ],
  "new_items_introduced": [
    "предмет который впервые появляется в этой сцене"
  ],
  "locations_mentioned": [
    {
      "name": "локация",
      "items_present": ["что находится в локации"],
      "access": "открыта|заперта|заперта изнутри"
    }
  ],
  "facts_stated": [
    "краткий факт (до 15 слов)"
  ],
  "relationships_mentioned": [
    {
      "person_a": "имя",
      "person_b": "имя",
      "relation": "тип отношений"
    }
  ]
}
"""

_SYSTEM_CONFLICT_CHECKER = """Ты — детектив-логик. Проверяешь новые факты против Story Bible.

Возвращай ТОЛЬКО валидный JSON без markdown.

{
  "conflicts": [
    {
      "type": "character_name|character_role|item_location|door_state|document_text|item_spawned|relationship|physical_impossibility",
      "description": "описание конфликта",
      "canonical_fact": "что в Story Bible",
      "scene_fact": "что в новой сцене",
      "severity": "critical|warning"
    }
  ],
  "is_consistent": true
}

Проверяй СТРОГО:

1. ПЕРСОНАЖИ: имя изменилось? роль/родство изменилось?
2. ДВЕРИ И ЗАМКИ: состояние двери противоречит зафиксированному?
   - Если в Bible "locked_from_outside", а в сцене "заперта изнутри" → CRITICAL
   - Если ключ был у персонажа А, а теперь у персонажа Б без объяснения → CRITICAL
3. ДОКУМЕНТЫ: текст письма/записки дополнен или изменён?
   - Если в Bible зафиксирован exact_text, а в сцене приводится другой/расширенный текст → CRITICAL
4. СПАВН ПРЕДМЕТОВ: в сцене появился новый ключевой предмет (сейф, оружие, украшение)?
   - Если предмет не был в scene_inventory ни одной предыдущей сцены → CRITICAL ("NO SPAWNING rule")
5. ПРЕДМЕТЫ: украденный предмет внезапно у другого персонажа?
6. МЁРТВЫЕ ДЕЙСТВУЮТ: мёртвый персонаж что-то делает?
7. ТЕЛЕПОРТАЦИЯ: персонаж в двух местах одновременно?

Severity:
- critical = грубое нарушение логики, сцена должна быть переписана
- warning = небольшая неточность, можно исправить при переписывании

НЕ придумывай конфликты. Если сомневаешься → is_consistent: true.
"""

_SYSTEM_LOGIC_VALIDATOR = """Ты — строгий валидатор логики детективного рассказа.

Получаешь текст сцены и JSON состояния мира (Story Bible).
Задача: найти ТОЛЬКО физические и логические невозможности.

Отвечай ТОЛЬКО в формате:
PASS — если логических нарушений нет
FAIL: [причина] — если есть нарушение

Проверяй:
1. Может ли дверь быть одновременно заперта изнутри и снаружи?
2. Может ли предмет находиться в двух местах?
3. Может ли мёртвый человек совершать действия?
4. Появился ли в сцене предмет которого не было в предыдущих сценах?
5. Изменился ли текст ранее зафиксированного документа?

Отвечай кратко — только PASS или FAIL с причиной.
"""

_SYSTEM_SYNOPSIS = """Ты — аналитик. Возвращай ТОЛЬКО валидный JSON без markdown.

Структура:
{
  "murderer": "имя убийцы",
  "victim": "имя жертвы",
  "motive": "мотив (кратко)",
  "method": "способ (кратко)",
  "stolen_item": "что украдено и где должно быть",
  "locked_room_solution": "механика запертой комнаты (кратко)"
}
"""


# ─────────────────────────── Story Bible Schema ───────────────────────────────


def _empty_story_bible() -> Dict[str, Any]:
    """Story Bible — канонический источник правды."""
    return {
        "synopsis": {
            "murderer": "",
            "victim": "",
            "motive": "",
            "method": "",
            "stolen_item": "",
            "locked_room_solution": "",
        },
        "characters": {},

        "items": {},

        "scene_inventory": {},

        "document_texts": {},
        "doors": {},
        "locations": {},
        "established_facts": [],
        "conflict_log": [],
        "scene_endings": {},

        "last_updated_scene": -1,
        "total_scenes": 0,
    }

# ─────────────────────────── Агент ────────────────────────────────────────────

class StateTrackerAgent:
    """
    Агент извлечения фактов и ведения Story Bible.
    Использует тот же DeepSeek API что и WriterAgent.
    """

    def __init__(
        self,
        api_key: str,
        folder_id: str,
        model_name: str = "deepseek-v4-flash",
        base_url: str = "https://llm.api.cloud.yandex.net/v1",
        auth_scheme: str = "Api-Key",
        timeout: int = 120,
        data_logging_enabled: bool = False,
    ):
        self.api_key = api_key
        self.folder_id = folder_id
        self.model_uri = f"gpt://{folder_id}/{model_name}"
        self.base_url = base_url.rstrip("/")
        self.auth_scheme = auth_scheme
        self.timeout = timeout
        self.data_logging_enabled = data_logging_enabled

        self._bible: Dict[str, Any] = _empty_story_bible()

        self.total_requests: int = 0
        self.total_conflicts_found: int = 0

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
        temperature: float = 0.1,
        max_tokens: int = 3000,
        max_retries: int = 3,
        retry_delay: float = 5.0,
    ) -> str:
        """
        Вызов DeepSeek API с retry.

        При finish_reason='length' возвращает частичный контент —
        _parse_json_response умеет восстанавливать обрезанные объекты.
        """
        last_error: Optional[Exception] = None
        current_max_tokens = max_tokens

        for attempt in range(1, max_retries + 1):
            try:
                payload = {
                    "model": self.model_uri,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": temperature,
                    "max_tokens": current_max_tokens,
                    "stream": False,
                }

                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=self._build_headers(),
                    json=payload,
                    timeout=self.timeout,
                )

                if not response.ok:
                    logger.error(
                        "StateTracker HTTP %d: %s",
                        response.status_code, response.text[:300],
                    )
                    response.raise_for_status()

                data = response.json()
                self.total_requests += 1

                choices = data.get("choices", [])
                if not choices:
                    raise RuntimeError("StateTracker: пустой choices в ответе")

                finish_reason = choices[0].get("finish_reason", "unknown")
                content = choices[0].get("message", {}).get("content", "")

                # Случай 1: нормальный ответ
                if finish_reason == "stop" and content and content.strip():
                    return content.strip()

                # Случай 2: ответ обрезан — возвращаем частичный контент
                if finish_reason == "length":
                    if content and content.strip():
                        logger.warning(
                            "StateTracker: ответ обрезан (length), "
                            "попытка починить JSON. "
                            "Попытка %d/%d, получено %d символов",
                            attempt, max_retries, len(content),
                        )
                        # Возвращаем как есть — fix_truncated_json починит
                        return content.strip()
                    else:
                        # content реально пустой — retry с большим лимитом
                        current_max_tokens = min(current_max_tokens + 1000, 5000)
                        logger.warning(
                            "StateTracker: length + пустой content, "
                            "увеличиваю max_tokens до %d. Попытка %d/%d",
                            current_max_tokens, attempt, max_retries,
                        )
                        raise RuntimeError(
                            "finish_reason=length + пустой content"
                        )

                # Случай 3: пустой контент по другой причине
                if not content or not content.strip():
                    raise RuntimeError(
                        f"StateTracker: пустой content "
                        f"(finish_reason={finish_reason!r})"
                    )

                return content.strip()

            except RuntimeError as exc:
                last_error = exc
                logger.warning(
                    "StateTracker._call_api попытка %d/%d: %s",
                    attempt, max_retries, exc,
                )
                if attempt < max_retries:
                    time.sleep(retry_delay)
            except Exception as exc:
                last_error = exc
                logger.error(
                    "StateTracker._call_api HTTP ошибка попытка %d/%d: %s",
                    attempt, max_retries, exc,
                )
                if attempt < max_retries:
                    time.sleep(retry_delay)

        raise RuntimeError(
            f"StateTracker: все {max_retries} попытки вернули ошибку. "
            f"Последняя: {last_error}"
        )

    def _parse_json_response(self, text: str) -> Dict[str, Any]:
        """
        Безопасно распарсить JSON из ответа модели.

        Умеет восстанавливать обрезанный JSON через fix_truncated_json.
        """
        if not text or not text.strip():
            return {}

        # Убираем markdown обёртки
        cleaned = re.sub(
            r"^```(?:json)?\s*", "", text.strip(), flags=re.MULTILINE
        )
        cleaned = re.sub(r"\s*```\s*$", "", cleaned, flags=re.MULTILINE)
        cleaned = cleaned.strip()

        # Стратегия 1: прямой парсинг
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # Стратегия 2: ищем первый { и берём всё от него
        start_idx = cleaned.find("{")
        if start_idx == -1:
            return {}
        cleaned = cleaned[start_idx:]

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # Стратегия 3: чиним обрезанный JSON
        # fix_truncated_json — свободная функция модуля (не метод класса!)
        fixed = fix_truncated_json(cleaned)
        try:
            result = json.loads(fixed)
            logger.debug(
                "StateTracker: JSON восстановлен из обрезанного ответа"
            )
            return result
        except json.JSONDecodeError:
            pass

        logger.warning(
            "StateTracker: не удалось распарсить JSON "
            "(первые 150 символов): %s",
            text[:150],
        )
        return {}

    # ── Инициализация ──────────────────────────────────────────────────────────

    def initialize_from_plan(
        self,
        plan: Dict[str, Any],
        initial_lore: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Инициализировать Story Bible из плана и начального лора.

        Вызывается ОДИН РАЗ после plan_story().

        Args:
            plan         : {title, scene_plan, ...}
            initial_lore : {past: {...}, present: {...}, future: {...}}

        Returns:
            Инициализированная Story Bible
        """
        logger.info("📚 StateTracker: Инициализирую Story Bible...")

        self._bible = _empty_story_bible()
        self._bible["total_scenes"] = len(plan.get("scene_plan", []))

        # ── Персонажи из лора ──────────────────────────────────────────────────
        for layer_name in ["past", "present"]:
            layer = initial_lore.get(layer_name, {})
            if not isinstance(layer, dict):
                continue
            chars = layer.get("characters", {})
            if not isinstance(chars, dict):
                continue

            for name, info in chars.items():
                if not isinstance(info, dict):
                    continue
                if name in self._bible["characters"]:
                    continue  # уже добавлен из предыдущего слоя

                role = info.get("role", "")
                role_lower = role.lower()
                is_investigator = any(
                    w in role_lower
                    for w in ["следователь", "инспектор", "детектив", "сыщик"]
                )

                self._bible["characters"][name] = {
                    "canonical_name": name,
                    "aliases": [],
                    "role": role,
                    "status": info.get("status", "alive"),
                    "location": info.get("location", ""),
                    "relationships": info.get("relationships", {}),
                    "traits": info.get("traits", []),
                    "is_suspect": not is_investigator,
                    "introduced_in_scene": 0,
                }

        # ── Улики как предметы ─────────────────────────────────────────────────
        for layer_name in ["past", "present"]:
            layer = initial_lore.get(layer_name, {})
            if not isinstance(layer, dict):
                continue
            clues = layer.get("clues", [])
            if not isinstance(clues, list):
                continue

            for clue in clues:
                if not isinstance(clue, dict):
                    continue
                clue_id = clue.get("id", "")
                if not clue_id or clue_id in self._bible["items"]:
                    continue

                self._bible["items"][clue_id] = {
                    "canonical_name": clue.get("description", clue_id),
                    "status": clue.get("status", "hidden"),
                    "location": clue.get("location", ""),
                    "holder": None,
                    "history": [
                        f"сцена 0: {clue.get('status', 'hidden')} "
                        f"в {clue.get('location', '?')}"
                    ],
                }

        # ── Синопсис через LLM ─────────────────────────────────────────────────
        self._extract_synopsis_from_lore(initial_lore, plan)

        suspects = [
            n for n, c in self._bible["characters"].items()
            if c.get("is_suspect", True)
        ]
        investigator = next(
            (
                n for n, c in self._bible["characters"].items()
                if not c.get("is_suspect", True)
            ),
            None,
        )

        logger.info(
            "📚 StateTracker: Bible инициализирована. "
            "Персонажей: %d, подозреваемых: %d, предметов: %d",
            len(self._bible["characters"]),
            len(suspects),
            len(self._bible["items"]),
        )
        logger.info("📚 Подозреваемые: %s", ", ".join(suspects))
        if investigator:
            logger.info("📚 Следователь: %s", investigator)

        return self._bible

    def _extract_synopsis_from_lore(
        self,
        initial_lore: Dict[str, Any],
        plan: Dict[str, Any],
    ) -> None:
        """Извлечь синопсис через LLM."""
        try:
            scene_goals = []
            for scene in plan.get("scene_plan", []):
                scene_goals.append(
                    f"Сцена {scene.get('index', 0) + 1}: "
                    f"{scene.get('goal', '')} | "
                    f"Событие: {scene.get('key_event', '')}"
                )

            future_secrets: List[str] = []
            future = initial_lore.get("future", {})
            if isinstance(future, dict):
                future_secrets = [
                    str(s) for s in future.get("secrets", [])
                ]

            prompt = (
                "Из плана сцен извлеки синопсис детектива.\n\n"
                "ПЛАН СЦЕН:\n"
                + "\n".join(scene_goals)
                + "\n\nСЕКРЕТЫ:\n"
                + "\n".join(future_secrets)
            )

            response = self._call_api(
                system_prompt=_SYSTEM_SYNOPSIS,
                user_prompt=prompt,
                temperature=0.1,
                max_tokens=3000,
            )
            synopsis = self._parse_json_response(response)
            if synopsis:
                self._bible["synopsis"].update(synopsis)
                logger.info(
                    "📚 Синопсис: убийца=%s, жертва=%s",
                    self._bible["synopsis"].get("murderer", "?"),
                    self._bible["synopsis"].get("victim", "?"),
                )
        except Exception as exc:
            logger.warning(
                "StateTracker: не удалось извлечь синопсис: %s", exc
            )

    # ── Обработка сцены ────────────────────────────────────────────────────────
    def process_scene(
        self,
        scene_text: str,
        scene_index: int,
        scene_title: str,
    ) -> Tuple[List[Dict[str, Any]], bool]:
        """Обработать написанную сцену."""
        logger.info(
            "📚 StateTracker: Обрабатываю сцену %d '%s'...",
            scene_index + 1, scene_title,
        )

        extracted = self._extract_facts(scene_text, scene_index)
        if not extracted:
            logger.warning(
                "StateTracker: не удалось извлечь факты из сцены %d",
                scene_index + 1,
            )
            # Даже без фактов сохраняем окончание сцены
            ending = _extract_scene_ending(scene_text, sentences=3)
            self._bible["scene_endings"][str(scene_index)] = ending
            return [], True

        conflicts = self._check_conflicts(extracted, scene_index)
        critical = [c for c in conflicts if c.get("severity") == "critical"]

        if not critical:
            # Передаём scene_text для сохранения окончания
            self._update_bible(
                extracted, scene_index, scene_title,
                scene_text=scene_text,
            )
        else:
            logger.warning(
                "📚 %d критических конфликтов — Bible НЕ обновляется",
                len(critical),
            )
            # Но окончание сцены всё равно сохраняем
            # (для плавного перехода в следующей сцене)
            ending = _extract_scene_ending(scene_text, sentences=3)
            self._bible["scene_endings"][str(scene_index)] = ending

        is_consistent = len(critical) == 0
        self.total_conflicts_found += len(conflicts)
        return conflicts, is_consistent
    
    # agents/state_tracker.py

    def get_scene_transition(self, prev_scene_index: int) -> str:
        """
        Получить окончание предыдущей сцены для плавного перехода.

        Args:
            prev_scene_index : Индекс предыдущей сцены (0-based)

        Returns:
            Текст для инжекции в промпт Писателя
        """
        ending = self._bible["scene_endings"].get(str(prev_scene_index), "")
        if not ending:
            return ""

        return (
            f"⚡ ПЛАВНЫЙ ПЕРЕХОД — продолжай СТРОГО с этого момента:\n"
            f"\"{ending}\"\n"
            f"НЕ описывай вход в комнату заново если персонаж уже там. "
            f"НЕ повторяй описание обстановки."
        )

    def get_location_state(self) -> str:
        """
        Получить текущее местоположение всех персонажей и предметов.

        Используется в промпте Писателя для проверки
        пространственной логики.
        """
        lines: List[str] = ["📍 ТЕКУЩИЕ КООРДИНАТЫ:"]

        chars = self._bible.get("characters", {})
        if chars:
            lines.append("  Персонажи:")
            for name, char in chars.items():
                loc = char.get("current_location") or char.get("location", "?")
                inventory = char.get("inventory", [])
                line = f"    • {char.get('canonical_name', name)}: {loc}"
                if inventory:
                    line += f" | при себе: {', '.join(inventory)}"
                lines.append(line)

        items = self._bible.get("items", {})
        if items:
            lines.append("  Предметы:")
            for key, item in items.items():
                loc = item.get("current_location") or item.get("location", "?")
                holder = item.get("holder")
                line = f"    • {item.get('canonical_name', key)}: {loc}"
                if holder:
                    line += f" (у {holder})"
                lines.append(line)

        docs = self._bible.get("document_texts", {})
        if docs:
            lines.append("  Документы:")
            for key, doc in docs.items():
                loc = doc.get("current_location") or doc.get("location", "?")
                lines.append(
                    f"    • {doc.get('canonical_name', key)}: {loc}"
                )

        return "\n".join(lines)

    def _extract_facts(
        self,
        scene_text: str,
        scene_index: int,
    ) -> Dict[str, Any]:
        """Извлечь структурированные факты из текста сцены."""
        try:
            scene_excerpt = scene_text[:800]
            if len(scene_text) > 800:
                scene_excerpt += "\n[сокращено]"

            prompt = (
                f"Сцена {scene_index + 1}:\n{scene_excerpt}\n\n"
                f"Извлеки факты. Особое внимание:\n"
                f"- doors_mentioned: состояние КАЖДОЙ двери упомянутой в сцене\n"
                f"- documents_mentioned: ТОЧНЫЙ текст если цитируется документ\n"
                f"- new_items_introduced: предметы которые ВПЕРВЫЕ появляются\n"
                f"Значения — не длиннее 10 слов каждое. Максимум 3 facts_stated."
            )

            response = self._call_api(
                system_prompt=_SYSTEM_EXTRACTOR,
                user_prompt=prompt,
                temperature=0.05,
                max_tokens=3000,
            )
            result = self._parse_json_response(response)
            if not result:
                logger.warning(
                    "StateTracker: пустой результат для сцены %d",
                    scene_index + 1,
                )
            return result
        except Exception as exc:
            logger.error("StateTracker._extract_facts: %s", exc)
            return {}

    

    def _check_conflicts(
        self,
        extracted: Dict[str, Any],
        scene_index: int,
    ) -> List[Dict[str, Any]]:
        """Проверить извлечённые факты против Story Bible."""
        if not self._bible["characters"] and not self._bible["items"]:
            return []

        all_conflicts: List[Dict[str, Any]] = []

        # ── Быстрые детерминированные проверки (без LLM) ──────────────────────
        fast_conflicts = self._fast_check(extracted, scene_index)
        all_conflicts.extend(fast_conflicts)

        # ── LLM-проверка для сложных случаев ──────────────────────────────────
        try:
            bible_summary = self._build_bible_summary_compact()
            extracted_compact = {
                "characters": [
                    {
                        "name": c.get("name", ""),
                        "role": c.get("role", ""),
                        "status": c.get("status", ""),
                    }
                    for c in extracted.get("characters_mentioned", [])
                ],
                "items": [
                    {
                        "name": i.get("name", ""),
                        "status": i.get("status", ""),
                        "holder": i.get("holder", ""),
                    }
                    for i in extracted.get("items_mentioned", [])
                ],
                "doors": extracted.get("doors_mentioned", []),
                "documents": [
                    {
                        "name": d.get("name", ""),
                        "exact_text": d.get("exact_text", "")[:100],
                    }
                    for d in extracted.get("documents_mentioned", [])
                ],
                "new_items": extracted.get("new_items_introduced", []),
                "relationships": extracted.get("relationships_mentioned", []),
            }

            prompt = (
                f"STORY BIBLE:\n{bible_summary}\n\n"
                f"ФАКТЫ ИЗ СЦЕНЫ {scene_index + 1}:\n"
                f"{json.dumps(extracted_compact, ensure_ascii=False, indent=2)}\n\n"
                f"Найди только ЯВНЫЕ конфликты. "
                f"Если сомневаешься — не добавляй конфликт."
            )
            response = self._call_api(
                system_prompt=_SYSTEM_CONFLICT_CHECKER,
                user_prompt=prompt,
                temperature=0.05,
                max_tokens=3000,
            )
            result = self._parse_json_response(response)
            llm_conflicts = result.get("conflicts", [])
            all_conflicts.extend(llm_conflicts)

        except Exception as exc:
            logger.error("StateTracker._check_conflicts LLM: %s", exc)

        # Логируем все конфликты
        if all_conflicts:
            logger.warning(
                "📚 StateTracker: найдено %d конфликтов в сцене %d",
                len(all_conflicts), scene_index + 1,
            )
            for c in all_conflicts:
                logger.warning(
                    "  [%s] %s | канон: '%s' | сцена: '%s'",
                    c.get("severity", "?"),
                    c.get("description", ""),
                    c.get("canonical_fact", ""),
                    c.get("scene_fact", ""),
                )
            self._bible["conflict_log"].extend(
                [{**c, "scene": scene_index + 1} for c in all_conflicts]
            )

        return all_conflicts

    def _fast_check(
        self,
        extracted: Dict[str, Any],
        scene_index: int,
    ) -> List[Dict[str, Any]]:
        """
        Быстрые детерминированные проверки без LLM.

        Проверяет:
        1. Состояние дверей против Bible
        2. Текст документов против зафиксированного
        3. NO SPAWNING rule — новые предметы в поздних сценах
        """
        conflicts: List[Dict[str, Any]] = []

        # ── Проверка 1: Двери ──────────────────────────────────────────────────
        for door_data in extracted.get("doors_mentioned", []):
            if not isinstance(door_data, dict):
                continue
            door_name = door_data.get("name", "").strip()
            if not door_name:
                continue

            door_key = door_name.lower().replace(" ", "_")[:30]
            if door_key not in self._bible["doors"]:
                continue  # новая дверь — конфликтов нет

            bible_door = self._bible["doors"][door_key]
            scene_status = door_data.get("locked_status", "")
            canonical_status = bible_door.get("locked_status", "")

            # Статусы противоположные по смыслу
            opposites = {
                "locked_from_outside": "locked_from_inside",
                "locked_from_inside": "locked_from_outside",
            }

            if (
                scene_status
                and canonical_status
                and scene_status != canonical_status
                and opposites.get(canonical_status) == scene_status
            ):
                conflicts.append({
                    "type": "door_state",
                    "description": (
                        f"Состояние двери '{door_name}' противоречит "
                        f"зафиксированному"
                    ),
                    "canonical_fact": (
                        f"дверь {canonical_status} "
                        f"(сцена {bible_door.get('established_in_scene', '?')})"
                    ),
                    "scene_fact": f"дверь {scene_status}",
                    "severity": "critical",
                })

            # Проверяем местонахождение ключа
            scene_key_holder = door_data.get("key_holder", "")
            canonical_key_holder = bible_door.get("key_holder", "")
            if (
                scene_key_holder
                and canonical_key_holder
                and scene_key_holder.lower() != canonical_key_holder.lower()
            ):
                conflicts.append({
                    "type": "item_location",
                    "description": (
                        f"Ключ от '{door_name}' телепортировался"
                    ),
                    "canonical_fact": (
                        f"ключ у {canonical_key_holder}"
                    ),
                    "scene_fact": f"ключ у {scene_key_holder}",
                    "severity": "critical",
                })

        # ── Проверка 2: Тексты документов ─────────────────────────────────────
        for doc_data in extracted.get("documents_mentioned", []):
            if not isinstance(doc_data, dict):
                continue
            doc_name = doc_data.get("name", "").strip()
            if not doc_name:
                continue

            doc_key = doc_name.lower().replace(" ", "_")[:30]
            if doc_key not in self._bible["document_texts"]:
                continue  # новый документ

            bible_doc = self._bible["document_texts"][doc_key]
            if not bible_doc.get("is_frozen"):
                continue

            canonical_text = bible_doc.get("exact_text", "")
            scene_text_fragment = doc_data.get("exact_text", "")

            if not canonical_text or not scene_text_fragment:
                continue

            # Проверяем: текст сцены начинается с канонического?
            # Если нет — возможно документ был изменён
            canon_start = canonical_text[:50].lower().strip()
            scene_start = scene_text_fragment[:50].lower().strip()

            if canon_start and scene_start and canon_start != scene_start:
                conflicts.append({
                    "type": "document_text",
                    "description": (
                        f"Текст документа '{doc_name}' изменён или "
                        f"дополнен задним числом"
                    ),
                    "canonical_fact": (
                        f"зафиксированный текст: '{canonical_text[:60]}...'"
                    ),
                    "scene_fact": (
                        f"текст в сцене: '{scene_text_fragment[:60]}...'"
                    ),
                    "severity": "critical",
                })

        # ── Проверка 3: NO SPAWNING rule ──────────────────────────────────────
        # Новые ключевые предметы в сцене 3+ без упоминания в сценах 1-2
        if scene_index >= 2:  # Проверяем начиная с 3-й сцены
            new_items = extracted.get("new_items_introduced", [])
            # Собираем все предметы из инвентаря первых двух сцен
            early_inventory: List[str] = []
            for loc_key, inv in self._bible["scene_inventory"].items():
                if inv.get("established_in_scene", 99) <= 2:
                    early_inventory.extend(
                        i.lower() for i in inv.get("items", [])
                    )
            # Добавляем известные предметы из Bible
            for item_key, item in self._bible["items"].items():
                early_inventory.append(
                    item.get("canonical_name", item_key).lower()
                )

            for new_item in new_items:
                if not new_item:
                    continue
                new_item_lower = new_item.lower()
                # Проверяем — есть ли похожий предмет в ранних сценах
                is_known = any(
                    new_item_lower in known or known in new_item_lower
                    for known in early_inventory
                )
                if not is_known:
                    conflicts.append({
                        "type": "item_spawned",
                        "description": (
                            f"Предмет '{new_item}' появился в сцене "
                            f"{scene_index + 1} без упоминания в сценах 1-2 "
                            f"(нарушение NO SPAWNING rule)"
                        ),
                        "canonical_fact": (
                            "предмет не упоминался в ранних сценах"
                        ),
                        "scene_fact": (
                            f"'{new_item}' введён как ключевой предмет"
                        ),
                        "severity": "critical",
                    })
        # ── Проверка 4: Location tracking — предмет доступен персонажу? ───────
        # Ищем в extracted паттерн "персонаж взаимодействует с предметом"
        for char_data in extracted.get("characters_mentioned", []):
            if not isinstance(char_data, dict):
                continue
            char_name = char_data.get("name", "")
            char_loc = char_data.get("location", "")
            actions = char_data.get("actions", [])

            if not char_loc or not actions:
                continue

            # Ищем упоминания предметов в действиях персонажа
            for action in actions:
                action_lower = action.lower()
                for item_key, item in self._bible.get("items", {}).items():
                    item_name = item.get("canonical_name", item_key).lower()
                    # Персонаж взаимодействует с предметом?
                    if item_name in action_lower or item_key.lower() in action_lower:
                        item_loc = (
                            item.get("current_location")
                            or item.get("location", "")
                        ).lower()
                        item_holder = (item.get("holder") or "").lower()
                        char_name_lower = char_name.lower()

                        # Предмет доступен если:
                        # 1. В той же локации что и персонаж
                        # 2. Или у самого персонажа (holder)
                        if (
                            item_loc
                            and char_loc
                            and item_loc not in char_loc.lower()
                            and char_loc.lower() not in item_loc
                            and char_name_lower not in item_holder
                        ):
                            conflicts.append({
                                "type": "item_location",
                                "description": (
                                    f"{char_name} взаимодействует с "
                                    f"'{item.get('canonical_name', item_key)}' "
                                    f"но находится в другой локации"
                                ),
                                "canonical_fact": (
                                    f"предмет в '{item_loc}', "
                                    f"персонаж в '{char_loc}'"
                                ),
                                "scene_fact": (
                                    f"действие: '{action}'"
                                ),
                                "severity": "warning",
                            })



        return conflicts

    # agents/state_tracker.py — в методе _update_bible

    def _update_bible(
        self,
        extracted: Dict[str, Any],
        scene_index: int,
        scene_title: str,
        scene_text: str = "",   # ← новый параметр для сохранения окончания
    ) -> None:
        """Обновить Story Bible новыми фактами из сцены."""

        # ── Персонажи ──────────────────────────────────────────────────────────
        for char_data in extracted.get("characters_mentioned", []):
            if not isinstance(char_data, dict):
                continue
            name = char_data.get("name", "").strip()
            if not name:
                continue
            canonical_key = self._find_canonical_character(name)

            if canonical_key:
                char = self._bible["characters"][canonical_key]
                # Обновляем current_location (было просто location)
                if char_data.get("location"):
                    char["current_location"] = char_data["location"]
                    char["location"] = char_data["location"]
                if char_data.get("status") and char_data["status"] != "unknown":
                    if char.get("status") != "dead":
                        char["status"] = char_data["status"]
                # Обновляем инвентарь если указан
                if char_data.get("inventory"):
                    char["inventory"] = char_data["inventory"]
            else:
                logger.info(
                    "📚 новый персонаж '%s' в сцене %d",
                    name, scene_index + 1,
                )
                self._bible["characters"][name] = {
                    "canonical_name": name,
                    "aliases": [],
                    "role": char_data.get("role", ""),
                    "status": char_data.get("status", "alive"),
                    "current_location": char_data.get("location", ""),
                    "location": char_data.get("location", ""),
                    "inventory": char_data.get("inventory", []),
                    "relationships": {},
                    "traits": [],
                    "is_suspect": True,
                    "introduced_in_scene": scene_index,
                }

        # ── Предметы ───────────────────────────────────────────────────────────
        for item_data in extracted.get("items_mentioned", []):
            if not isinstance(item_data, dict):
                continue
            item_name = item_data.get("name", "").strip().lower()
            if not item_name:
                continue
            item_key = self._find_canonical_item(item_name)
            if item_key:
                item = self._bible["items"][item_key]
                old_status = item.get("status")
                new_status = item_data.get("status")
                if new_status and new_status != old_status:
                    item["history"].append(
                        f"сцена {scene_index + 1}: {new_status} "
                        f"(было: {old_status})"
                    )
                    item["status"] = new_status
                # Обновляем current_location
                if item_data.get("location"):
                    item["current_location"] = item_data["location"]
                    item["location"] = item_data["location"]
                if item_data.get("holder"):
                    item["holder"] = item_data["holder"]
                    # Если у персонажа — location совпадает с его локацией
                    holder_key = self._find_canonical_character(
                        item_data["holder"]
                    )
                    if holder_key:
                        holder_loc = self._bible["characters"][holder_key].get(
                            "current_location", ""
                        )
                        if holder_loc:
                            item["current_location"] = holder_loc

        # ── Документы ──────────────────────────────────────────────────────────
        for doc_data in extracted.get("documents_mentioned", []):
            if not isinstance(doc_data, dict):
                continue
            doc_name = doc_data.get("name", "").strip()
            if not doc_name:
                continue
            doc_key = doc_name.lower().replace(" ", "_")[:30]
            exact_text = doc_data.get("exact_text", "").strip()

            if doc_key not in self._bible["document_texts"]:
                if exact_text:
                    self._bible["document_texts"][doc_key] = {
                        "canonical_name": doc_name,
                        "exact_text": exact_text,
                        "status": doc_data.get("status", "unknown"),
                        "current_location": doc_data.get("location", ""),
                        "location": doc_data.get("location", ""),
                        "established_in_scene": scene_index + 1,
                        "is_frozen": True,
                    }
                    logger.info(
                        "📚 Документ '%s' зафиксирован: '%s...'",
                        doc_name, exact_text[:50],
                    )
            else:
                doc = self._bible["document_texts"][doc_key]
                if doc_data.get("location"):
                    doc["current_location"] = doc_data["location"]
                    doc["location"] = doc_data["location"]

        # ── Двери ──────────────────────────────────────────────────────────────
        for door_data in extracted.get("doors_mentioned", []):
            if not isinstance(door_data, dict):
                continue
            door_name = door_data.get("name", "").strip()
            if not door_name:
                continue
            door_key = door_name.lower().replace(" ", "_")[:30]
            locked_status = door_data.get("locked_status", "")
            key_location = door_data.get("key_location", "")
            key_holder = door_data.get("key_holder", "")

            if door_key not in self._bible["doors"]:
                self._bible["doors"][door_key] = {
                    "canonical_name": door_name,
                    "locked_status": locked_status,
                    "key_location": key_location,
                    "key_holder": key_holder,
                    "established_in_scene": scene_index + 1,
                    "change_log": [{
                        "scene": scene_index + 1,
                        "status": locked_status,
                        "key_location": key_location,
                    }],
                }
                logger.info(
                    "📚 Дверь '%s': %s, ключ: %s",
                    door_name, locked_status, key_location,
                )
            else:
                door = self._bible["doors"][door_key]
                old_status = door.get("locked_status")
                if locked_status and locked_status != old_status:
                    logger.info(
                        "📚 Дверь '%s': %s → %s",
                        door_name, old_status, locked_status,
                    )
                    door["locked_status"] = locked_status
                    door["change_log"].append({
                        "scene": scene_index + 1,
                        "status": locked_status,
                        "key_location": key_location,
                    })
                if key_location:
                    door["key_location"] = key_location
                if key_holder:
                    door["key_holder"] = key_holder

        # ── Инвентарь локаций ──────────────────────────────────────────────────
        for loc_data in extracted.get("locations_mentioned", []):
            if not isinstance(loc_data, dict):
                continue
            loc_name = loc_data.get("name", "").strip()
            if not loc_name:
                continue
            loc_key = loc_name.lower().replace(" ", "_")
            if loc_key not in self._bible["scene_inventory"]:
                self._bible["scene_inventory"][loc_key] = {
                    "canonical_name": loc_name,
                    "items": [],
                    "established_in_scene": scene_index + 1,
                }
            inventory = self._bible["scene_inventory"][loc_key]
            for item in loc_data.get("items_present", []):
                if item and item not in inventory["items"]:
                    inventory["items"].append(item)
            if loc_key not in self._bible["locations"]:
                self._bible["locations"][loc_key] = {
                    "canonical_name": loc_name,
                    "connections": {},
                    "access_state": loc_data.get("access", "открыта"),
                }
            elif loc_data.get("access"):
                self._bible["locations"][loc_key]["access_state"] = (
                    loc_data["access"]
                )

        # ── Факты ─────────────────────────────────────────────────────────────
        for fact in extracted.get("facts_stated", []):
            self._bible["established_facts"].append({
                "scene": scene_index + 1,
                "fact": str(fact)[:100],
            })
        if len(self._bible["established_facts"]) > 30:
            self._bible["established_facts"] = (
                self._bible["established_facts"][-30:]
            )

        # ── Родство ────────────────────────────────────────────────────────────
        for rel in extracted.get("relationships_mentioned", []):
            if not isinstance(rel, dict):
                continue
            person_a = rel.get("person_a", "")
            person_b = rel.get("person_b", "")
            relation = rel.get("relation", "")
            if not all([person_a, person_b, relation]):
                continue
            key_a = self._find_canonical_character(person_a)
            if key_a:
                char = self._bible["characters"][key_a]
                if person_b not in char.get("relationships", {}):
                    char.setdefault("relationships", {})[person_b] = relation

        # ── НОВОЕ: Сохраняем окончание сцены для плавного перехода ────────────
        if scene_text:
            ending = _extract_scene_ending(scene_text, sentences=3)
            self._bible["scene_endings"][str(scene_index)] = ending
            logger.info(
                "📚 Окончание сцены %d сохранено: '%s...'",
                scene_index + 1, ending[:60],
            )

        self._bible["last_updated_scene"] = scene_index
        logger.info(
            "📚 Bible обновлена после сцены %d "
            "(двери: %d, документы: %d, персонажей: %d)",
            scene_index + 1,
            len(self._bible["doors"]),
            len(self._bible["document_texts"]),
            len(self._bible["characters"]),
        )


    # ── Поиск ─────────────────────────────────────────────────────────────────

    def _find_canonical_character(self, name: str) -> Optional[str]:
        """Найти канонический ключ персонажа (нечёткий поиск)."""
        name_lower = name.lower().strip()

        for key in self._bible["characters"]:
            if key.lower() == name_lower:
                return key

        for key in self._bible["characters"]:
            if name_lower in key.lower() or key.lower() in name_lower:
                return key

        for key, char in self._bible["characters"].items():
            for alias in char.get("aliases", []):
                if alias.lower() == name_lower:
                    return key

        return None

    def _find_canonical_item(self, name: str) -> Optional[str]:
        """Найти канонический ключ предмета."""
        name_lower = name.lower()
        for key, item in self._bible["items"].items():
            canon = item.get("canonical_name", "").lower()
            if (
                name_lower in canon
                or canon in name_lower
                or key in name_lower
                or name_lower in key
            ):
                return key
        return None

    # ── Форматирование Bible ───────────────────────────────────────────────────

    def _build_bible_summary_compact(self) -> str:
        """Компактное резюме для conflict checking внутри StateTracker."""
        lines: List[str] = []

        chars = self._bible.get("characters", {})
        if chars:
            suspects = [
                n for n, c in chars.items() if c.get("is_suspect", True)
            ]
            investigators = [
                n for n, c in chars.items()
                if not c.get("is_suspect", True)
            ]
            lines.append(
                f"Подозреваемые ({len(suspects)}): {', '.join(suspects)}"
            )
            if investigators:
                lines.append(f"Следователь: {', '.join(investigators)}")
            lines.append("")

            for name, char in chars.items():
                line = f"• {char.get('canonical_name', name)}"
                if char.get("role"):
                    line += f" [{char['role']}]"
                line += f" — {char.get('status', '?')}"
                if char.get("relationships"):
                    rels = "; ".join(
                        f"{k}:{v}"
                        for k, v in list(char["relationships"].items())[:2]
                    )
                    line += f" | {rels}"
                lines.append(line)

        items = self._bible.get("items", {})
        if items:
            lines.append("\nПредметы:")
            for key, item in items.items():
                line = (
                    f"• {item.get('canonical_name', key)}: "
                    f"{item.get('status', '?')}"
                )
                if item.get("holder"):
                    line += f" (у {item['holder']})"
                lines.append(line)

        return "\n".join(lines) if lines else "Bible пуста"

    def get_bible_for_writer(self) -> str:
        """Story Bible для инжекции в промпт Писателя."""
        lines: List[str] = [
            "╔══════════════════════════════════════╗",
            "║     STORY BIBLE — СТРОГО СЛЕДУЙ      ║",
            "╚══════════════════════════════════════╝",
            "",
        ]

        # ── Персонажи ──────────────────────────────────────────────────────────
        chars = self._bible.get("characters", {})
        if chars:
            suspects = [
                n for n, c in chars.items() if c.get("is_suspect", True)
            ]
            investigators = [
                n for n, c in chars.items()
                if not c.get("is_suspect", True)
            ]
            lines.append("ПЕРСОНАЖИ (имена и роли нельзя менять!):")
            lines.append(
                f"  Подозреваемых ровно {len(suspects)}: "
                f"{', '.join(suspects)}"
            )
            if investigators:
                lines.append(
                    f"  Следователь: {', '.join(investigators)}"
                )
            lines.append("")
            for name, char in chars.items():
                line = f"  • {char.get('canonical_name', name)}"
                if char.get("role"):
                    line += f" ({char['role']})"
                if char.get("status") == "dead":
                    line += " [МЁРТВ]"
                if char.get("location"):
                    line += f" — сейчас: {char['location']}"
                if char.get("relationships"):
                    rels = "; ".join(
                        f"{k} → {v}"
                        for k, v in list(char["relationships"].items())[:3]
                    )
                    line += f" | родство: {rels}"
                lines.append(line)
            lines.append("")

        # ── Предметы ───────────────────────────────────────────────────────────
        items = self._bible.get("items", {})
        if items:
            lines.append("ПРЕДМЕТЫ:")
            for key, item in items.items():
                line = (
                    f"  • {item.get('canonical_name', key)}: "
                    f"{item.get('status', '?')}"
                )
                if item.get("location"):
                    line += f" — {item['location']}"
                if item.get("holder"):
                    line += f" (держит: {item['holder']})"
                history = item.get("history", [])
                if len(history) > 1:
                    line += f" | история: {' → '.join(history[-2:])}"
                lines.append(line)
            lines.append("")

        # ── НОВОЕ: Документы с точным текстом ─────────────────────────────────
        docs = self._bible.get("document_texts", {})
        if docs:
            lines.append(
                "ДОКУМЕНТЫ (текст заморожен — нельзя менять или дополнять!):"
            )
            for key, doc in docs.items():
                lines.append(
                    f"  • {doc.get('canonical_name', key)} "
                    f"[сцена {doc.get('established_in_scene', '?')}]:"
                )
                if doc.get("exact_text"):
                    lines.append(
                        f"    ТОЧНЫЙ ТЕКСТ: \"{doc['exact_text']}\""
                    )
                lines.append(
                    f"    Статус: {doc.get('status', '?')}, "
                    f"место: {doc.get('location', '?')}"
                )
                if doc.get("is_frozen"):
                    lines.append(
                        "    ⚠️  ТЕКСТ ЗАМОРОЖЕН — нельзя дописывать или "
                        "изменять содержание этого документа!"
                    )
            lines.append("")

        # ── НОВОЕ: Состояние дверей ────────────────────────────────────────────
        doors = self._bible.get("doors", {})
        if doors:
            lines.append(
                "ДВЕРИ И ЗАМКИ (состояние заморожено — "
                "нельзя противоречить!):"
            )
            for key, door in doors.items():
                status = door.get("locked_status", "?")
                key_loc = door.get("key_location", "?")
                lines.append(
                    f"  • {door.get('canonical_name', key)}: "
                    f"{status}, ключ: {key_loc}"
                )
                # История изменений
                change_log = door.get("change_log", [])
                if len(change_log) > 1:
                    changes = " → ".join(
                        f"сц.{c['scene']}: {c['status']}"
                        for c in change_log[-3:]
                    )
                    lines.append(f"    История: {changes}")
            lines.append("")

        # ── НОВОЕ: NO SPAWNING rule ────────────────────────────────────────────
        scene_inv = self._bible.get("scene_inventory", {})
        all_known_items: List[str] = []
        for loc_data in scene_inv.values():
            all_known_items.extend(loc_data.get("items", []))
        for item in items.values():
            canon = item.get("canonical_name", "")
            if canon and canon not in all_known_items:
                all_known_items.append(canon)

        if all_known_items:
            lines.append(
                "⛔ NO SPAWNING RULE — можно использовать ТОЛЬКО "
                "эти предметы:"
            )
            for item_name in all_known_items[:15]:
                lines.append(f"  • {item_name}")
            lines.append(
                "  Нельзя вводить новые сейфы, оружие, украшения, "
                "документы если их не было в предыдущих сценах!"
            )
            lines.append("")

        # ── Факты ─────────────────────────────────────────────────────────────
        facts = self._bible.get("established_facts", [])
        if facts:
            lines.append("ФАКТЫ (не противоречь им!):")
            for f in facts[-8:]:
                lines.append(
                    f"  • [сц.{f.get('scene', '?')}] {f.get('fact', '')}"
                )

        return "\n".join(lines)

    def get_bible_for_critic(self) -> str:
        """Story Bible для Критика — включает синопсис."""
        lines: List[str] = []

        synopsis = self._bible.get("synopsis", {})
        if any(synopsis.values()):
            lines.append("=== СИНОПСИС (ТОЛЬКО ДЛЯ ПРОВЕРКИ) ===")
            if synopsis.get("murderer"):
                lines.append(f"Убийца: {synopsis['murderer']}")
            if synopsis.get("victim"):
                lines.append(f"Жертва: {synopsis['victim']}")
            if synopsis.get("stolen_item"):
                lines.append(
                    f"Украденный предмет: {synopsis['stolen_item']}"
                )
            if synopsis.get("locked_room_solution"):
                lines.append(
                    f"Механика комнаты: {synopsis['locked_room_solution']}"
                )
            lines.append("")

        lines.append(self.get_bible_for_writer())
        return "\n".join(lines)

    def get_conflicts_summary(self) -> str:
        """Краткое резюме всех найденных конфликтов."""
        conflicts = self._bible.get("conflict_log", [])
        if not conflicts:
            return "Конфликтов не обнаружено ✅"
        lines = [f"Всего конфликтов: {len(conflicts)}"]
        for c in conflicts[-5:]:
            lines.append(
                f"  [сцена {c.get('scene', '?')} | "
                f"{c.get('severity', '?')}] "
                f"{c.get('description', '')}"
            )
        return "\n".join(lines)

    def get_stats(self) -> Dict[str, int]:
        return {
            "total_requests": self.total_requests,
            "total_conflicts_found": self.total_conflicts_found,
        }
    
    def validate_logic(
        self,
        scene_text: str,
        scene_index: int,
    ) -> Tuple[bool, str]:
        """
        Двухэтапный валидатор логики (без GigaChat).

        Этап 1: Детерминированные правила (мгновенно, без LLM)
        Этап 2: LLM-проверка физической возможности

        Args:
            scene_text  : Текст сцены
            scene_index : Индекс сцены

        Returns:
            Tuple[passed (bool), reason (str)]
        """
        # ── Этап 1: Детерминированные правила ─────────────────────────────────

        # Проверяем текст сцены на упоминание зафиксированных документов
        for doc_key, doc in self._bible.get("document_texts", {}).items():
            if not doc.get("is_frozen"):
                continue
            canonical_text = doc.get("exact_text", "")
            if not canonical_text or len(canonical_text) < 20:
                continue

            # Ищем в тексте сцены цитаты из этого документа
            # Берём первые 30 символов как "fingerprint"
            fingerprint = canonical_text[:30].lower()
            if fingerprint in scene_text.lower():
                # Документ цитируется — проверяем что текст не расширен
                # Ищем контекст вокруг цитаты
                idx = scene_text.lower().find(fingerprint)
                if idx >= 0:
                    # Берём 200 символов после начала цитаты
                    cited_fragment = scene_text[idx: idx + 200]
                    # Если цитата длиннее канонической — возможно дописали
                    if len(cited_fragment) > len(canonical_text) + 50:
                        logger.warning(
                            "📚 LogicValidator: документ '%s' возможно дополнен. "
                            "Канон: %d символов, в сцене: %d+ символов",
                            doc.get("canonical_name", doc_key),
                            len(canonical_text),
                            len(cited_fragment),
                        )

        # ── Этап 2: LLM проверка физических невозможностей ────────────────────
        try:
            # Компактное состояние для проверки
            state_for_validator = {
                "doors": {
                    k: {
                        "status": v.get("locked_status"),
                        "key_at": v.get("key_location"),
                    }
                    for k, v in self._bible.get("doors", {}).items()
                },
                "documents": {
                    k: {
                        "exact_text": v.get("exact_text", "")[:80],
                        "is_frozen": v.get("is_frozen"),
                    }
                    for k, v in self._bible.get("document_texts", {}).items()
                },
                "known_items": list(self._bible.get("items", {}).keys()),
                "scene_inventory_scene_1_2": [
                    item
                    for loc_data in self._bible.get(
                        "scene_inventory", {}
                    ).values()
                    if loc_data.get("established_in_scene", 99) <= 2
                    for item in loc_data.get("items", [])
                ],
            }

            prompt = (
                f"ТЕКУЩЕЕ СОСТОЯНИЕ МИРА:\n"
                f"{json.dumps(state_for_validator, ensure_ascii=False, indent=2)}\n\n"
                f"ТЕКСТ СЦЕНЫ {scene_index + 1}:\n{scene_text[:600]}\n\n"
                f"Проверь: есть ли физические или логические невозможности?"
            )

            response = self._call_api(
                system_prompt=_SYSTEM_LOGIC_VALIDATOR,
                user_prompt=prompt,
                temperature=0.05,
                max_tokens=3000,
            )

            response = response.strip()
            if response.upper().startswith("FAIL"):
                reason = response[4:].strip(" :")
                logger.warning(
                    "📚 LogicValidator: FAIL в сцене %d — %s",
                    scene_index + 1, reason,
                )
                return False, reason

            return True, "PASS"

        except Exception as exc:
            logger.error("StateTracker.validate_logic: %s", exc)
            return True, "PASS (ошибка валидатора — пропускаем)"

def _extract_scene_ending(text: str, sentences: int = 3) -> str:
    """
    Извлечь последние N предложений из текста сцены.

    Используется для инжекции в промпт следующей сцены
    чтобы обеспечить плавный переход без повторений.

    Args:
        text      : Полный текст сцены
        sentences : Количество предложений для извлечения

    Returns:
        Строка с последними N предложениями
    """
    if not text:
        return ""

    # Разбиваем на предложения по точке/восклицанию/вопросу
    parts = re.split(r"(?<=[.!?…])\s+", text.strip())
    # Убираем пустые
    parts = [p.strip() for p in parts if p.strip()]

    if not parts:
        return text[-200:]

    last_sentences = parts[-sentences:]
    return " ".join(last_sentences)