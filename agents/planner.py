"""
Агент-Планировщик сцены (Scene Planner).

Реализует двухэтапную генерацию:
    Этап 1: Planner создаёт короткий логический план сцены (1 абзац)
    Этап 2: Critic_Logic проверяет план (быстро, ~500 токенов)
    Этап 3: Writer разворачивает одобренный план в художественный текст

Это предотвращает ситуацию когда Writer пишет 500 слов
а Critic находит логическую дыру и всё приходится переписывать.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)


_SYSTEM_SCENE_PLANNER = """Ты — сценарист детективного рассказа.
Пишешь короткий логический план одной сцены (не художественный текст!).

Формат ответа — ОДИН абзац, 50-100 слов:
«[Персонаж] делает [действие] в [локации]. Он/она обнаруживает/узнаёт [факт/улику].
Это приводит к [следствие]. Сцена заканчивается [крючок для следующей сцены].»

Правила:
- СТРОГО следуй Story Bible (персонажи, предметы, их местонахождение)
- Все предметы должны быть в правильных локациях
- Нельзя вводить новые предметы не упомянутые ранее
- Нельзя описывать действия "за кадром" — только то что происходит сейчас
- Убийца НЕ раскрывается до финальной сцены
"""

_SYSTEM_PLAN_VALIDATOR = """Ты — логический валидатор плана сцены детектива.

Получаешь короткий план сцены и Story Bible.
Проверяешь ТОЛЬКО логику и соответствие фактам.

Отвечай строго в одном из форматов:
APPROVED — план логически корректен
REJECTED: [причина] — план содержит логическую ошибку

Проверяй:
1. Персонажи в правильных локациях?
2. Предметы доступны персонажам которые с ними взаимодействуют?
3. Нет новых предметов которых не было раньше?
4. Нет действий "за кадром"?
5. Не раскрывается убийца раньше финала?

Будь строгим но справедливым. Отвечай кратко.
"""


class ScenePlannerAgent:
    """
    Агент-Планировщик сцены.

    Создаёт короткий логический план сцены перед тем как
    Writer начнёт писать художественный текст.
    Это экономит токены и итерации.
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

        self.total_requests: int = 0
        self.total_plan_approvals: int = 0
        self.total_plan_rejections: int = 0

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
        temperature: float = 0.3,
        max_tokens: int = 3000,
    ) -> str:
        """
        Вызов API для планировщика.

        Args:
            system_prompt : Системный промпт
            user_prompt   : Пользовательский промпт
            temperature   : Температура генерации
            max_tokens    : Максимальное число токенов

        Returns:
            Текстовый ответ модели

        Raises:
            RuntimeError : Если ответ пустой или choices пустой
            requests.HTTPError : При HTTP-ошибке
        """
        payload = {
            "model": self.model_uri,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
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
                "ScenePlanner HTTP %d: %s",
                response.status_code,
                response.text[:300],
            )
            response.raise_for_status()

        data = response.json()
        self.total_requests += 1

        choices = data.get("choices", [])
        if not choices:
            raise RuntimeError("ScenePlanner: пустой choices в ответе API")

        content = choices[0].get("message", {}).get("content", "")
        if not content or not content.strip():
            finish_reason = choices[0].get("finish_reason", "unknown")
            raise RuntimeError(
                f"ScenePlanner: пустой content "
                f"(finish_reason={finish_reason!r})"
            )

        return content.strip()

    def create_scene_plan(
        self,
        scene_plan: Dict[str, Any],
        story_bible: str,
        location_state: str,
        scene_transition: str,
        approved_scenes: List[Dict[str, Any]],
    ) -> Tuple[str, bool]:
        """
        Создать и валидировать короткий логический план сцены.

        Двухэтапный процесс:
            1. Planner создаёт план (~50-100 слов)
            2. Validator проверяет план (~500 токенов, быстро)

        Если план отклонён — переписываем (максимум 3 попытки).
        Если все попытки исчерпаны — возвращаем последний вариант плана
        с флагом False (не одобрен), чтобы Writer всё равно мог работать.

        Args:
            scene_plan      : {index, title, goal, key_event, location}
            story_bible     : Текущая Story Bible
            location_state  : Текущие координаты персонажей/предметов
            scene_transition: Окончание предыдущей сцены
            approved_scenes : Уже написанные сцены

        Returns:
            Tuple[текст плана, был ли одобрен валидатором]
            Текст плана может быть пустой строкой только если все попытки
            завершились исключением до первого успешного вызова API.
        """
        scene_index = scene_plan.get("index", 0)
        scene_title = scene_plan.get("title", "")

        logger.info(
            "🗺️  ScenePlanner: Создаю план сцены %d '%s'...",
            scene_index + 1,
            scene_title,
        )

        # Краткое резюме предыдущей сцены для контекста
        prev_summary = ""
        if approved_scenes:
            last = approved_scenes[-1]
            last_text = last.get("text", "")
            prev_summary = (
                f"Предыдущая сцена '{last.get('title', '?')}': "
                f"{last_text[:150]}{'...' if len(last_text) > 150 else ''}"
            )

        # Храним последний успешно сгенерированный план между попытками.
        # Инициализируем пустой строкой — если все попытки упадут до
        # первого успешного _call_api, вернём "".
        last_plan_text: str = ""

        # Причина последнего отклонения — добавляется в промпт следующей попытки
        last_rejection_reason: str = ""

        for attempt in range(1, 4):  # максимум 3 попытки
            try:
                # ── Этап 1: Создаём план ───────────────────────────────────────

                # Если предыдущий план был отклонён — добавляем причину,
                # чтобы модель не повторяла ту же ошибку
                rejection_hint = ""
                if last_rejection_reason:
                    rejection_hint = (
                        f"\nПРЕДЫДУЩИЙ ПЛАН БЫЛ ОТКЛОНЁН: {last_rejection_reason}\n"
                        f"Исправь эту ошибку в новом плане.\n"
                    )

                plan_prompt = (
                    f"{story_bible}\n\n"
                    f"{location_state}\n\n"
                    + (f"{scene_transition}\n\n" if scene_transition else "")
                    + f"СЦЕНА ДЛЯ ПЛАНИРОВАНИЯ:\n"
                    f"Название: {scene_title}\n"
                    f"Цель: {scene_plan.get('goal', '')}\n"
                    f"Ключевое событие: {scene_plan.get('key_event', '')}\n"
                    f"Место: {scene_plan.get('location', '')}\n\n"
                    + (f"КОНТЕКСТ:\n{prev_summary}\n\n" if prev_summary else "")
                    + rejection_hint
                    + "Напиши логический план этой сцены (50-100 слов). "
                    "Только логика, без художественного текста."
                )

                current_plan_text = self._call_api(
                    system_prompt=_SYSTEM_SCENE_PLANNER,
                    user_prompt=plan_prompt,
                    temperature=0.3,
                    max_tokens=3000,
                )

                # Сохраняем — теперь есть что вернуть даже при провале валидации
                last_plan_text = current_plan_text

                logger.info(
                    "🗺️  ScenePlanner: план создан (попытка %d): '%s...'",
                    attempt,
                    current_plan_text[:80],
                )

                # ── Этап 2: Валидируем план ────────────────────────────────────

                validate_prompt = (
                    f"STORY BIBLE (компактно):\n"
                    f"{story_bible[:800]}\n\n"
                    f"{location_state}\n\n"
                    f"ПЛАН СЦЕНЫ {scene_index + 1}:\n{current_plan_text}\n\n"
                    f"Проверь план. Отвечай: APPROVED или REJECTED: причина"
                )

                validation = self._call_api(
                    system_prompt=_SYSTEM_PLAN_VALIDATOR,
                    user_prompt=validate_prompt,
                    temperature=0.05,
                    max_tokens=3000,
                )

                validation = validation.strip()

                if validation.upper().startswith("APPROVED"):
                    self.total_plan_approvals += 1
                    logger.info(
                        "🗺️  ScenePlanner: план APPROVED на попытке %d",
                        attempt,
                    )
                    return current_plan_text, True

                # План отклонён — извлекаем причину для следующей итерации
                # Формат ответа: "REJECTED: причина" или "REJECTED причина"
                rejected_prefix = validation[:8].upper()  # "REJECTED"
                if rejected_prefix.startswith("REJECTED"):
                    last_rejection_reason = validation[8:].strip(" :")
                else:
                    last_rejection_reason = validation

                self.total_plan_rejections += 1
                logger.warning(
                    "🗺️  ScenePlanner: план REJECTED (попытка %d): %s",
                    attempt,
                    last_rejection_reason,
                )

                if attempt < 3:
                    time.sleep(2)

            except Exception as exc:
                logger.error(
                    "ScenePlanner попытка %d: %s", attempt, exc
                )
                if attempt < 3:
                    time.sleep(3)
                # last_plan_text не обновляем — сохраняем лучший из предыдущих

        # ── Все попытки исчерпаны ──────────────────────────────────────────────
        # Возвращаем последний сгенерированный план (может быть пустым если
        # все три попытки упали с исключением до первого успешного вызова API)
        if last_plan_text:
            logger.warning(
                "🗺️  ScenePlanner: план не одобрен за 3 попытки, "
                "используем последний вариант: '%s...'",
                last_plan_text[:60],
            )
        else:
            logger.error(
                "🗺️  ScenePlanner: все 3 попытки завершились ошибкой, "
                "план пустой — Writer будет работать без плана"
            )

        return last_plan_text, False

    def get_stats(self) -> Dict[str, int]:
        """Вернуть статистику работы планировщика."""
        return {
            "total_requests": self.total_requests,
            "total_plan_approvals": self.total_plan_approvals,
            "total_plan_rejections": self.total_plan_rejections,
        }