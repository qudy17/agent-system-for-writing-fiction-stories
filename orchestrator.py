"""
Оркестратор автономной мультиагентной детективной системы.

Архитектура:
    ScenePlannerAgent (DeepSeek V4) — планирует логику сцены
        ↕
    WriterAgent (DeepSeek V4) — пишет художественный текст
        ↕
    StateTrackerAgent (DeepSeek V4) — ведёт Story Bible
        ↕
    CriticAgent (GigaChat) — проверяет качество
        ↕
    OntologyMemoryMCP (stdio JSON-RPC) — хранит состояние

Алгоритм для каждой сцены:
    1. ScenePlanner создаёт короткий логический план (50-100 слов)
    2. Validator проверяет план (~500 токенов, быстро)
    3. Writer разворачивает план в художественный текст
    4. StateTracker проверяет факты и конфликты
    5. LogicValidator проверяет физические невозможности
    6. Critic делает финальную проверку
    7. При APPROVE — обновляем Bible и переходим к следующей сцене
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar

from dotenv import load_dotenv

# ─────────────────────────── Пути и конфигурация ──────────────────────────────

PROJECT_DIR = Path(__file__).resolve().parent
MCP_SERVER_PATH = PROJECT_DIR / "mcp" / "memory_server.py"

load_dotenv(PROJECT_DIR / ".env")

# ─────────────────────────── Логирование ──────────────────────────────────────

def _setup_logging() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s │ %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    return logging.getLogger("orchestrator")


logger = _setup_logging()

# ─────────────────────────── Константы ────────────────────────────────────────

MAX_ITERATIONS_PER_SCENE: int = 5
RETRY_DELAY: float = 5.0
MAX_API_RETRIES: int = 3
STORY_DIR = Path(os.getenv("STORY_DIR", str(PROJECT_DIR / "stories")))

T = TypeVar("T")


# ─────────────────────────── Загрузка конфигурации ────────────────────────────

def _load_config() -> Dict[str, Any]:
    """
    Загрузить и валидировать конфигурацию из переменных окружения.

    Raises:
        SystemExit: Если обязательные переменные не заданы.
    """
    errors: List[str] = []

    yandex_api_key = os.getenv("YANDEX_API_KEY", "")
    yandex_folder_id = os.getenv("YANDEX_FOLDER_ID", "")
    yandex_model = os.getenv("YANDEX_MODEL_NAME", "deepseek-v4-flash")
    yandex_base_url = os.getenv(
        "YANDEX_BASE_URL", "https://llm.api.cloud.yandex.net/v1"
    )
    yandex_auth_scheme = os.getenv("YANDEX_AUTH_SCHEME", "Api-Key")
    yandex_temperature = float(os.getenv("YANDEX_TEMPERATURE", "0.7"))
    yandex_max_tokens = int(os.getenv("YANDEX_MAX_TOKENS", "4000"))
    yandex_timeout = int(os.getenv("YANDEX_TIMEOUT", "120"))
    data_logging = (
        os.getenv("YANDEX_DATA_LOGGING_ENABLED", "false").lower() == "true"
    )

    gigachat_credentials = os.getenv("GIGACHAT_CREDENTIALS", "")
    gigachat_scope = os.getenv("GIGACHAT_SCOPE", "GIGACHAT_API_PERS")
    gigachat_model = os.getenv("GIGACHAT_MODEL", "GigaChat")

    if not yandex_api_key:
        errors.append("YANDEX_API_KEY не задан в .env")
    if not yandex_folder_id:
        errors.append("YANDEX_FOLDER_ID не задан в .env")
    if not gigachat_credentials:
        errors.append("GIGACHAT_CREDENTIALS не задан в .env")

    if errors:
        logger.error("❌ Ошибки конфигурации:")
        for err in errors:
            logger.error("   • %s", err)
        sys.exit(1)

    return {
        "yandex_api_key": yandex_api_key,
        "yandex_folder_id": yandex_folder_id,
        "yandex_model": yandex_model,
        "yandex_base_url": yandex_base_url,
        "yandex_auth_scheme": yandex_auth_scheme,
        "yandex_temperature": yandex_temperature,
        "yandex_max_tokens": yandex_max_tokens,
        "yandex_timeout": yandex_timeout,
        "data_logging": data_logging,
        "gigachat_credentials": gigachat_credentials,
        "gigachat_scope": gigachat_scope,
        "gigachat_model": gigachat_model,
    }


# ─────────────────────────── Retry-обёртка ────────────────────────────────────

def _retry(
    fn: Callable[[], T],
    *,
    max_attempts: int = MAX_API_RETRIES,
    delay: float = RETRY_DELAY,
    label: str = "операция",
) -> T:
    """
    Выполнить функцию с повторными попытками при исключении.

    Args:
        fn           : Callable без аргументов (используй lambda)
        max_attempts : Максимальное число попыток
        delay        : Пауза между попытками в секундах
        label        : Метка для логов

    Returns:
        Результат fn() при успехе

    Raises:
        Exception: Последнее пойманное исключение после исчерпания попыток
    """
    last_exc: Optional[Exception] = None

    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            logger.warning(
                "⚠️  %s: попытка %d/%d не удалась: %s",
                label, attempt, max_attempts, exc,
            )
            if attempt < max_attempts:
                logger.info("   ⏳ Жду %.0f сек...", delay)
                time.sleep(delay)

    raise last_exc  # type: ignore[misc]


# ─────────────────────────── Сохранение рассказа ──────────────────────────────

def _save_story(
    title: str,
    approved_scenes: List[Dict[str, Any]],
    memory: Dict[str, Any],
    writer_stats: Dict[str, int],
    critic_stats: Dict[str, int],
    tracker_stats: Dict[str, int],
    planner_stats: Dict[str, int],
    elapsed_seconds: float,
) -> Path:
    """
    Сохранить итоговый рассказ в markdown-файл.

    Args:
        title           : Название рассказа
        approved_scenes : Список одобренных сцен
        memory          : Финальное состояние памяти MCP
        writer_stats    : Статистика WriterAgent
        critic_stats    : Статистика CriticAgent
        tracker_stats   : Статистика StateTrackerAgent
        planner_stats   : Статистика ScenePlannerAgent
        elapsed_seconds : Общее время работы

    Returns:
        Путь к сохранённому файлу.
    """
    STORY_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_title = "".join(
        c if (c.isalnum() or c in " _-") else "_" for c in title
    )[:50].strip("_ ")
    filename = STORY_DIR / f"{timestamp}_{safe_title}.md"

    lines: List[str] = [
        f"# {title}",
        "",
        f"*Создано: {datetime.now().strftime('%d.%m.%Y %H:%M')}*  ",
        f"*Писатель: DeepSeek V4 | Критик: GigaChat*",
        "",
        "---",
        "",
    ]

    for scene in approved_scenes:
        lines.append(f"## {scene.get('title', 'Сцена')}")
        lines.append("")
        lines.append(scene.get("text", ""))
        lines.append("")
        lines.append("---")
        lines.append("")

    meta = memory.get("meta") or {}
    lines += [
        "## Мета-информация",
        "",
        "| Параметр | Значение |",
        "|---|---|",
        f"| Сцен написано | {len(approved_scenes)} |",
        f"| Общее время | {elapsed_seconds / 60:.1f} мин |",
        f"| Writer: запросов | {writer_stats.get('total_requests', 0)} |",
        f"| Writer: токенов | {writer_stats.get('total_tokens', 0)} |",
        f"| Critic: запросов | {critic_stats.get('total_requests', 0)} |",
        f"| Critic: одобрений | {critic_stats.get('total_approvals', 0)} |",
        f"| Critic: отказов | {critic_stats.get('total_rejections', 0)} |",
        f"| StateTracker: запросов | {tracker_stats.get('total_requests', 0)} |",
        f"| StateTracker: конфликтов | {tracker_stats.get('total_conflicts_found', 0)} |",
        f"| ScenePlanner: запросов | {planner_stats.get('total_requests', 0)} |",
        f"| ScenePlanner: планов ✅ | {planner_stats.get('total_plan_approvals', 0)} |",
        f"| ScenePlanner: планов ❌ | {planner_stats.get('total_plan_rejections', 0)} |",
        f"| Дата создания | {meta.get('created_at', 'N/A')} |",
        "",
    ]

    filename.write_text("\n".join(lines), encoding="utf-8")
    logger.info("💾 Рассказ сохранён: %s", filename)
    return filename


# ─────────────────────────── Главная функция ──────────────────────────────────

def run_autonomous_agency(prompt: str) -> Optional[Path]:
    """
    Запустить автономный цикл мультиагентной детективной системы.

    Args:
        prompt : Стартовая идея для детективного рассказа

    Returns:
        Путь к файлу с готовым рассказом, или None при критической ошибке.
    """
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    config = _load_config()
    story_start_time = time.time()

    logger.info("=" * 70)
    logger.info("🕵️  ДЕТЕКТИВНОЕ АГЕНТСТВО: Автономная мультиагентная система")
    logger.info("=" * 70)
    logger.info("📝 Стартовая идея: %s", prompt[:120])
    logger.info("=" * 70)

    # ── Импорты агентов ────────────────────────────────────────────────────────
    from agents.writer import WriterAgent
    from agents.critic import CriticAgent
    from agents.state_tracker import StateTrackerAgent
    from agents.planner import ScenePlannerAgent
    from mcp.memory_client import OntologyMemoryMCP

    # ── Создаём агентов ────────────────────────────────────────────────────────
    writer = WriterAgent(
        api_key=config["yandex_api_key"],
        folder_id=config["yandex_folder_id"],
        model_name=config["yandex_model"],
        base_url=config["yandex_base_url"],
        auth_scheme=config["yandex_auth_scheme"],
        temperature=config["yandex_temperature"],
        max_tokens=config["yandex_max_tokens"],
        timeout=config["yandex_timeout"],
        data_logging_enabled=config["data_logging"],
    )

    critic = CriticAgent(
        credentials=config["gigachat_credentials"],
        scope=config["gigachat_scope"],
        model=config["gigachat_model"],
    )

    tracker = StateTrackerAgent(
        api_key=config["yandex_api_key"],
        folder_id=config["yandex_folder_id"],
        model_name=config["yandex_model"],
        base_url=config["yandex_base_url"],
        auth_scheme=config["yandex_auth_scheme"],
        timeout=config["yandex_timeout"],
        data_logging_enabled=config["data_logging"],
    )

    planner = ScenePlannerAgent(
        api_key=config["yandex_api_key"],
        folder_id=config["yandex_folder_id"],
        model_name=config["yandex_model"],
        base_url=config["yandex_base_url"],
        auth_scheme=config["yandex_auth_scheme"],
        timeout=config["yandex_timeout"],
        data_logging_enabled=config["data_logging"],
    )

    memory = OntologyMemoryMCP(
        server_path=MCP_SERVER_PATH,
        story_dir=STORY_DIR,
    )

    logger.info("🧠 Запускаю MCP memory_server...")
    memory.start()
    memory.reset()
    logger.info("🧠 MCP готов. Память очищена.")

    try:
        return _run_story_loop(
            prompt=prompt,
            writer=writer,
            critic=critic,
            tracker=tracker,
            planner=planner,
            memory=memory,
            story_start_time=story_start_time,
            config=config,
        )
    except KeyboardInterrupt:
        logger.info("\n⛔ Прервано пользователем")
        return None
    except Exception as exc:
        logger.exception("❌ Критическая ошибка оркестратора: %s", exc)
        return None
    finally:
        memory.close()
        logger.info("🧠 MCP memory_server остановлен")


# ─────────────────────────── Основной цикл ────────────────────────────────────

def _run_story_loop(
    prompt: str,
    writer: Any,
    critic: Any,
    tracker: Any,
    planner: Any,
    memory: Any,
    story_start_time: float,
    config: Dict[str, Any],
) -> Optional[Path]:
    """
    Основной цикл генерации истории.

    Фазы:
        1. Планирование (Writer.plan_story)
        2. Инициализация Story Bible (StateTracker)
        3. Цикл сцен: Plan → Write → Track → Review → Approve
        4. Финализация и сохранение

    Returns:
        Путь к сохранённому файлу или None при ошибке.
    """

    # ══════════════════════════════════════════════════════════════════════════
    # ФАЗА 1: Планирование истории
    # ══════════════════════════════════════════════════════════════════════════

    logger.info("\n" + "─" * 70)
    logger.info("📖 ФАЗА 1: Писатель планирует историю...")
    logger.info("─" * 70)

    plan = _retry(
        lambda: writer.plan_story(prompt),
        label="Writer.plan_story",
    )

    title: str = plan.get("title", "Детективная история")
    scene_plan: List[Dict[str, Any]] = plan.get("scene_plan") or []
    initial_lore: Dict[str, Any] = plan.get("initial_lore") or {}

    if not scene_plan:
        logger.error("❌ Писатель не создал план сцен!")
        return None

    logger.info("📖 Название: «%s»", title)
    logger.info("📖 Сцен запланировано: %d", len(scene_plan))
    for scene in scene_plan:
        logger.info(
            "   %d. %s — %s",
            scene.get("index", 0) + 1,
            scene.get("title", "?"),
            scene.get("goal", ""),
        )

    # Сохраняем план и лор в MCP
    memory.initialize(
        title=title,
        initial_lore=initial_lore,
        scene_plan=scene_plan,
    )
    logger.info("🧠 MCP: Лор и план сохранены")

    # ══════════════════════════════════════════════════════════════════════════
    # ФАЗА 2: Инициализация Story Bible
    # ══════════════════════════════════════════════════════════════════════════

    logger.info("\n" + "─" * 70)
    logger.info("📚 ФАЗА 2: Инициализация Story Bible...")
    logger.info("─" * 70)

    try:
        bible = _retry(
            lambda: tracker.initialize_from_plan(plan, initial_lore),
            label="StateTracker.initialize_from_plan",
        )

        chars = bible.get("characters", {})
        items = bible.get("items", {})
        suspects = [n for n, c in chars.items() if c.get("is_suspect", True)]
        investigators = [
            n for n, c in chars.items() if not c.get("is_suspect", True)
        ]

        logger.info("📚 Story Bible готова:")
        logger.info("   Персонажей: %d", len(chars))
        logger.info(
            "   Подозреваемых: %d → %s",
            len(suspects),
            ", ".join(suspects) if suspects else "не определены",
        )
        logger.info(
            "   Следователь: %s",
            ", ".join(investigators) if investigators else "не определён",
        )
        logger.info("   Предметов: %d", len(items))

        synopsis = bible.get("synopsis", {})
        if synopsis.get("murderer"):
            logger.info("   🔐 Убийца (секрет): %s", synopsis["murderer"])
        if synopsis.get("locked_room_solution"):
            logger.info(
                "   🔐 Механика комнаты: %s",
                synopsis["locked_room_solution"][:80],
            )

    except Exception as exc:
        logger.error(
            "❌ Не удалось инициализировать Story Bible: %s. "
            "Продолжаем без неё.",
            exc,
        )

    # ══════════════════════════════════════════════════════════════════════════
    # ФАЗА 3: Основной цикл сцен
    # ══════════════════════════════════════════════════════════════════════════

    approved_scenes: List[Dict[str, Any]] = []
    story_finished = False

    while not story_finished:

        # ── Читаем текущий индекс из MCP ──────────────────────────────────────
        full_memory = memory.get_full_memory()
        meta = full_memory.get("meta") or {}
        current_scene_index: int = meta.get("current_scene_index", 0)
        story_finished = meta.get("story_finished", False)

        if story_finished:
            logger.info("✅ MCP сообщил: история завершена")
            break

        if current_scene_index >= len(scene_plan):
            logger.info("✅ Все %d сцен написаны", len(scene_plan))
            break

        current_scene_plan = scene_plan[current_scene_index]

        logger.info("\n" + "─" * 70)
        logger.info(
            "🎬 СЦЕНА %d / %d: «%s»",
            current_scene_index + 1,
            len(scene_plan),
            current_scene_plan.get("title", "?"),
        )
        logger.info("─" * 70)

        # Читаем контекст памяти MCP
        memory_context = memory.get_context("all")

        # ── Цикл написания и проверки одной сцены ─────────────────────────────
        scene_text, scene_diff, scene_approved = _write_and_review_scene(
            writer=writer,
            critic=critic,
            tracker=tracker,
            planner=planner,
            scene_plan=current_scene_plan,
            memory_context=memory_context,
            approved_scenes=approved_scenes,
        )

        # ── Сохраняем результат ────────────────────────────────────────────────
        if scene_text:
            # Сохраняем в MCP
            memory.add_approved_scene(
                scene_index=current_scene_index,
                title=current_scene_plan.get(
                    "title", f"Сцена {current_scene_index + 1}"
                ),
                text=scene_text,
            )

            approved_scenes.append({
                "index": current_scene_index,
                "title": current_scene_plan.get(
                    "title", f"Сцена {current_scene_index + 1}"
                ),
                "text": scene_text,
                "approved": scene_approved,
            })

            # Обновляем MCP-память
            scene_diff["advance_scene_index"] = True
            update_result = memory.update_state(scene_diff)

            # Логируем конфликты MCP
            mcp_conflicts = update_result.get("conflicts", [])
            if mcp_conflicts:
                logger.warning(
                    "⚠️  MCP конфликты памяти в сцене %d:",
                    current_scene_index + 1,
                )
                for conflict in mcp_conflicts:
                    logger.warning("   🔴 %s", conflict)

            logger.info(
                "🧠 MCP обновлён. Индекс: %d, Улик: %d",
                update_result.get("current_scene_index", "?"),
                update_result.get("clue_registry_size", 0),
            )
            story_finished = update_result.get("story_finished", False)

        else:
            # Сцена не написана — принудительно переходим
            logger.error(
                "❌ Сцена %d не написана. Переходим дальше.",
                current_scene_index + 1,
            )
            memory.update_state({"advance_scene_index": True})

    # ══════════════════════════════════════════════════════════════════════════
    # ФАЗА 4: Финализация
    # ══════════════════════════════════════════════════════════════════════════

    logger.info("\n" + "=" * 70)
    logger.info("🏁 ИСТОРИЯ ЗАВЕРШЕНА!")
    logger.info("=" * 70)

    if not approved_scenes:
        logger.error("❌ Ни одна сцена не была написана")
        return None

    elapsed = time.time() - story_start_time
    final_memory = memory.get_full_memory()
    writer_stats = writer.get_stats()
    critic_stats = critic.get_stats()
    tracker_stats = tracker.get_stats()
    planner_stats = planner.get_stats()

    # Финальный отчёт Story Bible
    logger.info("\n📚 STORY BIBLE — финальный отчёт конфликтов:")
    logger.info(tracker.get_conflicts_summary())

    story_path = _save_story(
        title=title,
        approved_scenes=approved_scenes,
        memory=final_memory,
        writer_stats=writer_stats,
        critic_stats=critic_stats,
        tracker_stats=tracker_stats,
        planner_stats=planner_stats,
        elapsed_seconds=elapsed,
    )

    # ── Итоговая статистика ────────────────────────────────────────────────────
    logger.info("\n📊 ИТОГОВАЯ СТАТИСТИКА:")
    logger.info("─" * 40)
    logger.info("⏱️  Время работы   : %.1f мин", elapsed / 60)
    logger.info(
        "📝 Сцен написано  : %d / %d",
        len(approved_scenes), len(scene_plan),
    )
    logger.info(
        "✍️  Writer         : %d запросов, %d токенов",
        writer_stats["total_requests"],
        writer_stats["total_tokens"],
    )
    logger.info(
        "🔍 Critic          : %d запросов, %d ✅, %d ❌",
        critic_stats["total_requests"],
        critic_stats["total_approvals"],
        critic_stats["total_rejections"],
    )
    logger.info(
        "📚 StateTracker    : %d запросов, %d конфликтов",
        tracker_stats["total_requests"],
        tracker_stats["total_conflicts_found"],
    )
    logger.info(
        "🗺️  ScenePlanner    : %d запросов, %d планов ✅, %d планов ❌",
        planner_stats["total_requests"],
        planner_stats["total_plan_approvals"],
        planner_stats["total_plan_rejections"],
    )
    logger.info("💾 Файл            : %s", story_path)
    logger.info("=" * 70)

    return story_path


# ─────────────────────────── Цикл одной сцены ─────────────────────────────────

def _write_and_review_scene(
    writer: Any,
    critic: Any,
    tracker: Any,
    planner: Any,
    scene_plan: Dict[str, Any],
    memory_context: Dict[str, Any],
    approved_scenes: List[Dict[str, Any]],
) -> Tuple[str, Dict[str, Any], bool]:
    """
    Полный цикл написания и проверки одной сцены.

    Порядок:
        0. ScenePlanner создаёт и валидирует логический план
        На каждой итерации:
            1. Writer пишет / переписывает по плану
            2. StateTracker проверяет факты (fast_check + LLM)
            3. LogicValidator проверяет физические невозможности
            4. Critic делает финальную проверку
            5. APPROVE → выходим | замечания → следующая итерация

    Args:
        writer          : WriterAgent
        critic          : CriticAgent
        tracker         : StateTrackerAgent
        planner         : ScenePlannerAgent
        scene_plan      : {index, title, goal, key_event, location}
        memory_context  : Полное состояние памяти из MCP
        approved_scenes : Уже одобренные сцены

    Returns:
        Tuple[текст сцены, diff для MCP, флаг одобрения]
    """
    scene_text: str = ""
    scene_diff: Dict[str, Any] = {}
    scene_approved: bool = False
    critic_feedback: str = ""
    tracker_conflicts: List[Dict[str, Any]] = []
    approved_plan: str = ""

    scene_index = scene_plan.get("index", 0)
    scene_title = scene_plan.get("title", f"Сцена {scene_index + 1}")

    # ── Загружаем Story Bible и вспомогательные данные ────────────────────────
    bible_for_writer = ""
    bible_for_critic = ""
    location_state = ""
    scene_transition = ""

    try:
        bible_for_writer = tracker.get_bible_for_writer()
        bible_for_critic = tracker.get_bible_for_critic()
        location_state = tracker.get_location_state()

        # Плавный переход из предыдущей сцены (начиная со 2-й)
        if scene_index > 0:
            scene_transition = tracker.get_scene_transition(scene_index - 1)
            if scene_transition:
                logger.info(
                    "  ⚡ Плавный переход из сцены %d загружен",
                    scene_index,
                )

        logger.info(
            "  📚 Story Bible: %d персонажей, %d предметов, "
            "%d документов, %d дверей",
            len(tracker._bible.get("characters", {})),
            len(tracker._bible.get("items", {})),
            len(tracker._bible.get("document_texts", {})),
            len(tracker._bible.get("doors", {})),
        )
    except Exception as exc:
        logger.warning(
            "  ⚠️  Не удалось загрузить Story Bible: %s", exc
        )

    # ══════════════════════════════════════════════════════════════════════════
    # ЭТАП 0: ScenePlanner создаёт логический план
    # ══════════════════════════════════════════════════════════════════════════

    logger.info(
        "\n  🗺️  ScenePlanner: создаю план сцены %d...",
        scene_index + 1,
    )

    try:
        all_characters = list(
            tracker._bible.get("characters", {}).keys()
        )
        
        approved_plan, plan_approved = planner.create_scene_plan(
            scene_plan=scene_plan,
            story_bible=bible_for_writer,
            location_state=location_state,
            scene_transition=scene_transition,
            approved_scenes=approved_scenes,
            all_characters=all_characters,
        )
        if approved_plan:
            status = "✅ одобрен" if plan_approved else "⚠️ без одобрения"
            logger.info(
                "  🗺️  План сцены %s: '%s...'",
                status, approved_plan[:80],
            )
        else:
            logger.warning("  🗺️  ScenePlanner вернул пустой план")
    except Exception as exc:
        logger.warning(
            "  ⚠️  ScenePlanner недоступен: %s. Продолжаем без плана.",
            exc,
        )
        approved_plan = ""

    # ══════════════════════════════════════════════════════════════════════════
    # ЭТАП 1-N: Цикл Writer ↔ StateTracker ↔ Critic
    # ══════════════════════════════════════════════════════════════════════════

    for iteration in range(1, MAX_ITERATIONS_PER_SCENE + 1):

        logger.info(
            "\n  ✍️  [Итерация %d / %d] Писатель пишет...",
            iteration, MAX_ITERATIONS_PER_SCENE,
        )

        # ── Шаг 1: Writer пишет или переписывает ──────────────────────────────
        try:
            if iteration == 1:
                # Первый черновик — используем одобренный план
                scene_text, scene_diff = _retry(
                    lambda: writer.write_scene(
                        scene_plan=scene_plan,
                        memory_context=memory_context,
                        approved_scenes=approved_scenes,
                        story_bible=bible_for_writer,
                        approved_plan=approved_plan,
                        scene_transition=scene_transition,
                    ),
                    label=f"Writer.write_scene (сцена {scene_index + 1})",
                )
            else:
                # Переписываем — делаем snapshot переменных для lambda
                _text_snap = scene_text
                _fb_snap = critic_feedback
                _iter_snap = iteration
                _conf_snap = tracker_conflicts

                scene_text, scene_diff = _retry(
                    lambda: writer.rewrite_scene(
                        original_text=_text_snap,
                        critic_feedback=_fb_snap,
                        scene_plan=scene_plan,
                        memory_context=memory_context,
                        iteration=_iter_snap,
                        story_bible=bible_for_writer,
                        conflicts=_conf_snap,
                    ),
                    label=f"Writer.rewrite_scene (итерация {iteration})",
                )

        except Exception as exc:
            logger.error(
                "  ❌ Writer не справился (итерация %d): %s",
                iteration, exc,
            )
            if scene_text:
                logger.warning(
                    "  ⚠️  Принимаем текст предыдущей итерации"
                )
                scene_approved = True
            break

        word_count = len(scene_text.split())
        logger.info(
            "  ✍️  Черновик готов: %d слов %s",
            word_count,
            "✅" if 300 <= word_count <= 550 else "⚠️ вне лимита",
        )

        # ── Шаг 2: StateTracker проверяет факты ───────────────────────────────
        logger.info("  📚 StateTracker проверяет факты...")
        try:
            tracker_conflicts, is_consistent = tracker.process_scene(
                scene_text=scene_text,
                scene_index=scene_index,
                scene_title=scene_title,
            )

            critical_conflicts = [
                c for c in tracker_conflicts
                if c.get("severity") == "critical"
            ]

            if critical_conflicts:
                logger.warning(
                    "  📚 Критических конфликтов: %d",
                    len(critical_conflicts),
                )
                for c in critical_conflicts:
                    logger.warning(
                        "    🔴 %s | канон: '%s' | в тексте: '%s'",
                        c.get("description", ""),
                        c.get("canonical_fact", ""),
                        c.get("scene_fact", ""),
                    )
            elif tracker_conflicts:
                logger.info(
                    "  📚 Предупреждений (warning): %d",
                    len(tracker_conflicts),
                )
            else:
                logger.info("  📚 StateTracker: конфликтов нет ✅")

        except Exception as exc:
            logger.warning(
                "  ⚠️  StateTracker недоступен (итерация %d): %s",
                iteration, exc,
            )
            tracker_conflicts = []

        # ── Шаг 3: LogicValidator — физические невозможности ──────────────────
        logger.info("  🔬 LogicValidator проверяет логику...")
        try:
            logic_passed, logic_reason = tracker.validate_logic(
                scene_text=scene_text,
                scene_index=scene_index,
            )
            if not logic_passed:
                logger.warning(
                    "  🔬 LogicValidator: FAIL — %s", logic_reason
                )
                # Добавляем как critical конфликт для Critic
                tracker_conflicts.append({
                    "type": "physical_impossibility",
                    "description": logic_reason,
                    "canonical_fact": "текущее состояние Story Bible",
                    "scene_fact": "текст сцены противоречит логике",
                    "severity": "critical",
                })
            else:
                logger.info("  🔬 LogicValidator: PASS ✅")
        except Exception as exc:
            logger.warning(
                "  ⚠️  LogicValidator недоступен (итерация %d): %s",
                iteration, exc,
            )

        # ── Шаг 4: Critic делает финальную проверку ───────────────────────────
        logger.info("  🔍 Критик проверяет сцену...")
        try:
            is_approved, critic_feedback = _retry(
                lambda: critic.review_scene(
                    scene_text=scene_text,
                    memory_context=memory_context,
                    scene_plan=scene_plan,
                    approved_scenes=approved_scenes,
                    story_bible=bible_for_critic,
                    tracker_conflicts=tracker_conflicts,
                ),
                label=f"Critic.review_scene (итерация {iteration})",
            )
        except Exception as exc:
            logger.error(
                "  ❌ Critic недоступен (итерация %d): %s. "
                "Принимаем черновик автоматически.",
                iteration, exc,
            )
            scene_approved = True
            break

        # ── Шаг 5: Обрабатываем решение ───────────────────────────────────────
        if is_approved:
            logger.info(
                "  ✅ [APPROVE] Сцена «%s» одобрена на итерации %d",
                scene_title, iteration,
            )
            scene_approved = True
            break

        # Критик нашёл замечания
        logger.info(
            "  ❌ Замечания (итерация %d / %d):\n     %s",
            iteration,
            MAX_ITERATIONS_PER_SCENE,
            critic_feedback.replace("\n", "\n     "),
        )

        if iteration < MAX_ITERATIONS_PER_SCENE:
            logger.info(
                "  ↩️  Отправляю на переработку (итерация %d)...",
                iteration + 1,
            )
        else:
            logger.warning(
                "  ⚠️  Лимит итераций (%d) исчерпан для «%s». "
                "Принимаем последний вариант.",
                MAX_ITERATIONS_PER_SCENE, scene_title,
            )
            scene_approved = True

    return scene_text, scene_diff, scene_approved


# ─────────────────────────── Точка входа ──────────────────────────────────────

if __name__ == "__main__":
    INITIAL_PROMPT = """
    Место действия: Петербург, 1913 год. Туманный ноябрьский вечер.

    В запертом кабинете на втором этаже особняка графа Васильева найден мёртвым
    известный ювелир Адольф Зимберг. Дверь была заперта изнутри. На столе —
    незаконченное письмо и пустой бокал из-под коньяка. Исчезло знаменитое
    «Синее сердце Каспия» — редкий сапфир стоимостью в целое состояние.

    Персонажи в доме в момент смерти:
    - Граф Евгений Васильев (хозяин дома, 58 лет, скрытый должник)
    - Елизавета Васильева (жена графа, 34 года, бывшая актриса)
    - Дмитрий Воронов (племянник графа, 26 лет, игрок)
    - Мадам Соле (личная секретарша, 45 лет, знает все тайны)
    - Афанасий (дворецкий, 60 лет, служит семье 30 лет)

    Создай запутанный, атмосферный детектив с неожиданным разоблачением.
    Все сюжетные повороты должны быть связаны и обоснованы.
    Максимальная длина рассказа 2500 слов.
    """.strip()

    result = run_autonomous_agency(INITIAL_PROMPT)

    if result:
        print(f"\n✅ Готово! Рассказ сохранён: {result}")
    else:
        print("\n❌ Не удалось создать рассказ")
        sys.exit(1)