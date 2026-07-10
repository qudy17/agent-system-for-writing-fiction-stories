"""
Оркестратор автономной мультиагентной детективной системы.

Точка входа: run_autonomous_agency(prompt)

Архитектура:
    WriterAgent (DeepSeek V4 via Yandex AI)
        ↕ (черновик + diff)
    CriticAgent (GigaChat)
        ↕ ([APPROVE] или замечания)
    OntologyMemoryMCP (stdio JSON-RPC, файл памяти)

Алгоритм:
    1. Писатель генерирует лор и план сцен → сохраняем в MCP
    2. Для каждой сцены:
       a. Писатель пишет черновик (читая память из MCP)
       b. Критик проверяет (читая память из MCP)
       c. Если замечания → Писатель переписывает (макс. MAX_ITERATIONS_PER_SCENE раз)
       d. При [APPROVE] → сохраняем сцену, обновляем память, идём к следующей
    3. После всех сцен → собираем итоговый рассказ и сохраняем в файл
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TypeVar

from dotenv import load_dotenv

# ─────────────────────────── Пути и конфигурация ──────────────────────────────

PROJECT_DIR = Path(__file__).resolve().parent
MCP_SERVER_PATH = PROJECT_DIR / "mcp" / "memory_server.py"

load_dotenv(PROJECT_DIR / ".env")

# ─────────────────────────── Логирование ──────────────────────────────────────

def _setup_logging() -> logging.Logger:
    """
    Настроить логирование.

    Формат: время │ сообщение
    Подавляем шумные логи от requests/urllib3.
    """
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

# Максимум итераций Писатель ↔ Критик на одну сцену
MAX_ITERATIONS_PER_SCENE: int = 5

# Пауза между повторными попытками при ошибках API (секунды)
RETRY_DELAY: float = 5.0

# Количество повторных попыток при ошибке одного API-вызова
MAX_API_RETRIES: int = 3

STORY_DIR = Path(os.getenv("STORY_DIR", str(PROJECT_DIR / "stories")))

# TypeVar для типизации _retry
T = TypeVar("T")


# ─────────────────────────── Загрузка конфигурации ────────────────────────────

def _load_config() -> Dict[str, Any]:
    """
    Загрузить и валидировать конфигурацию из переменных окружения.

    Raises:
        SystemExit: Если обязательные переменные не заданы.
    """
    errors: List[str] = []

    # ── Yandex AI (DeepSeek) ───────────────────────────────────────────────────
    yandex_api_key = os.getenv("YANDEX_API_KEY", "")
    yandex_folder_id = os.getenv("YANDEX_FOLDER_ID", "")
    yandex_model = os.getenv("YANDEX_MODEL_NAME", "deepseek-v4-flash")
    yandex_base_url = os.getenv(
        "YANDEX_BASE_URL", "https://llm.api.cloud.yandex.net/v1"
    )
    yandex_auth_scheme = os.getenv("YANDEX_AUTH_SCHEME", "Api-Key")
    yandex_temperature = float(os.getenv("YANDEX_TEMPERATURE", "0.7"))
    # Увеличенный дефолт — сцены длинные, нужен запас
    yandex_max_tokens = int(os.getenv("YANDEX_MAX_TOKENS", "4000"))
    yandex_timeout = int(os.getenv("YANDEX_TIMEOUT", "120"))
    data_logging = (
        os.getenv("YANDEX_DATA_LOGGING_ENABLED", "false").lower() == "true"
    )

    # ── GigaChat ───────────────────────────────────────────────────────────────
    gigachat_credentials = os.getenv("GIGACHAT_CREDENTIALS", "")
    gigachat_scope = os.getenv("GIGACHAT_SCOPE", "GIGACHAT_API_PERS")
    gigachat_model = os.getenv("GIGACHAT_MODEL", "GigaChat")

    # ── Валидация обязательных полей ──────────────────────────────────────────
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
                logger.info(
                    "   ⏳ Жду %.0f сек перед следующей попыткой...", delay
                )
                time.sleep(delay)

    # Все попытки исчерпаны
    raise last_exc  # type: ignore[misc]


# ─────────────────────────── Сохранение рассказа ──────────────────────────────

def _save_story(
    title: str,
    approved_scenes: List[Dict[str, Any]],
    memory: Dict[str, Any],
    writer_stats: Dict[str, int],
    critic_stats: Dict[str, int],
    elapsed_seconds: float,
) -> Path:
    """
    Сохранить итоговый рассказ в markdown-файл.

    Args:
        title           : Название рассказа
        approved_scenes : Список одобренных сцен
        memory          : Финальное состояние памяти
        writer_stats    : Статистика писателя
        critic_stats    : Статистика критика
        elapsed_seconds : Общее время работы в секундах

    Returns:
        Путь к сохранённому файлу.
    """
    STORY_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Безопасное имя файла: только буквы, цифры, пробелы, дефисы, подчёркивания
    safe_title = "".join(
        c if (c.isalnum() or c in " _-") else "_" for c in title
    )[:50].strip("_ ")
    filename = STORY_DIR / f"{timestamp}_{safe_title}.md"

    lines: List[str] = [
        f"# {title}",
        "",
        f"*Создано: {datetime.now().strftime('%d.%m.%Y %H:%M')}*  ",
        f"*Писатель: DeepSeek V4 (Yandex AI) | Критик: GigaChat*",
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

    # ── Статистика в конце файла ───────────────────────────────────────────────
    meta = memory.get("meta") or {}
    lines += [
        "## Мета-информация",
        "",
        f"| Параметр | Значение |",
        f"|---|---|",
        f"| Сцен написано | {len(approved_scenes)} |",
        f"| Общее время | {elapsed_seconds / 60:.1f} мин |",
        f"| Writer: запросов | {writer_stats.get('total_requests', 0)} |",
        f"| Writer: токенов | {writer_stats.get('total_tokens', 0)} |",
        f"| Critic: запросов | {critic_stats.get('total_requests', 0)} |",
        f"| Critic: одобрений | {critic_stats.get('total_approvals', 0)} |",
        f"| Critic: отказов | {critic_stats.get('total_rejections', 0)} |",
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

    Агенты взаимодействуют без участия человека до тех пор,
    пока все сцены не будут написаны и одобрены Критиком.

    Args:
        prompt : Стартовая идея для детективного рассказа

    Returns:
        Путь к файлу с готовым рассказом, или None при критической ошибке.
    """
    # ── Подавляем предупреждения urllib3 об SSL (нужно для GigaChat) ──────────
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    config = _load_config()
    story_start_time = time.time()

    logger.info("=" * 70)
    logger.info("🕵️  ДЕТЕКТИВНОЕ АГЕНТСТВО: Автономная мультиагентная система")
    logger.info("=" * 70)
    logger.info("📝 Стартовая идея: %s", prompt[:120])
    logger.info("=" * 70)

    # ── Создаём агентов ────────────────────────────────────────────────────────
    from agents.writer import WriterAgent
    from agents.critic import CriticAgent
    from mcp.memory_client import OntologyMemoryMCP

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

    memory = OntologyMemoryMCP(
        server_path=MCP_SERVER_PATH,
        story_dir=STORY_DIR,
    )

    # ── Запускаем MCP-сервер памяти ────────────────────────────────────────────
    logger.info("🧠 Запускаю MCP memory_server...")
    memory.start()
    memory.reset()
    logger.info("🧠 MCP готов. Память очищена.")

    try:
        return _run_story_loop(
            prompt=prompt,
            writer=writer,
            critic=critic,
            memory=memory,
            story_start_time=story_start_time,
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


def _run_story_loop(
    prompt: str,
    writer: Any,
    critic: Any,
    memory: Any,
    story_start_time: float,
) -> Optional[Path]:
    """
    Основной цикл генерации истории.

    Вынесен из run_autonomous_agency для чистоты структуры.
    try/finally с закрытием MCP остаётся в вызывающей функции.

    Returns:
        Путь к сохранённому файлу рассказа.
    """
    # ── ФАЗА 1: Планирование ───────────────────────────────────────────────────
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

    # ── ФАЗА 2: Основной цикл сцен ────────────────────────────────────────────
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

        # Читаем полный контекст памяти для агентов
        memory_context = memory.get_context("all")

        # ── Цикл написания и проверки одной сцены ─────────────────────────────
        scene_text, scene_diff, scene_approved = _write_and_review_scene(
            writer=writer,
            critic=critic,
            scene_plan=current_scene_plan,
            memory_context=memory_context,
            approved_scenes=approved_scenes,
        )

        # ── Сохраняем результат и обновляем память ─────────────────────────────
        if scene_text:
            memory.add_approved_scene(
                scene_index=current_scene_index,
                title=current_scene_plan.get("title", f"Сцена {current_scene_index + 1}"),
                text=scene_text,
            )

            approved_scenes.append({
                "index": current_scene_index,
                "title": current_scene_plan.get("title", f"Сцена {current_scene_index + 1}"),
                "text": scene_text,
                "approved": scene_approved,
            })

            # Гарантируем advance_scene_index=True перед обновлением
            scene_diff["advance_scene_index"] = True
            update_result = memory.update_state(scene_diff)

            logger.info(
                "🧠 MCP обновлён. Текущий индекс сцены: %d",
                update_result.get("current_scene_index", "?"),
            )
            story_finished = update_result.get("story_finished", False)

        else:
            # Сцена не написана — форсируем переход, чтобы не застрять в цикле
            logger.error(
                "❌ Сцена %d не написана. Принудительно переходим дальше.",
                current_scene_index + 1,
            )
            memory.update_state({"advance_scene_index": True})

    # ── ФАЗА 3: Финализация ────────────────────────────────────────────────────
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

    story_path = _save_story(
        title=title,
        approved_scenes=approved_scenes,
        memory=final_memory,
        writer_stats=writer_stats,
        critic_stats=critic_stats,
        elapsed_seconds=elapsed,
    )

    # ── Итоговая статистика в консоль ─────────────────────────────────────────
    logger.info("\n📊 ИТОГОВАЯ СТАТИСТИКА:")
    logger.info("─" * 40)
    logger.info("⏱️  Время работы : %.1f мин", elapsed / 60)
    logger.info(
        "📝 Сцен написано: %d / %d",
        len(approved_scenes), len(scene_plan),
    )
    logger.info(
        "✍️  Writer       : %d запросов, %d токенов",
        writer_stats["total_requests"],
        writer_stats["total_tokens"],
    )
    logger.info(
        "🔍 Critic        : %d запросов, %d ✅ одобрено, %d ❌ отклонено",
        critic_stats["total_requests"],
        critic_stats["total_approvals"],
        critic_stats["total_rejections"],
    )
    logger.info("💾 Файл          : %s", story_path)
    logger.info("=" * 70)

    return story_path


def _write_and_review_scene(
    writer: Any,
    critic: Any,
    scene_plan: Dict[str, Any],
    memory_context: Dict[str, Any],
    approved_scenes: List[Dict[str, Any]],
) -> tuple[str, Dict[str, Any], bool]:
    """
    Выполнить полный цикл написания и проверки одной сцены.

    Писатель пишет черновик → Критик проверяет → если замечания,
    Писатель переписывает. Повторяем до MAX_ITERATIONS_PER_SCENE раз.

    Args:
        writer          : WriterAgent
        critic          : CriticAgent
        scene_plan      : План текущей сцены
        memory_context  : Текущее состояние памяти (все слои)
        approved_scenes : Уже одобренные сцены

    Returns:
        Tuple[текст сцены, diff, флаг одобрения]
        - Если сцена не написана вообще: ("", {}, False)
    """
    scene_text: str = ""
    scene_diff: Dict[str, Any] = {}
    scene_approved: bool = False
    # Инициализируем заранее — используется в rewrite на 2-й+ итерации
    critic_feedback: str = ""

    for iteration in range(1, MAX_ITERATIONS_PER_SCENE + 1):
        logger.info(
            "\n  ✍️  [Итерация %d / %d] Писатель пишет...",
            iteration, MAX_ITERATIONS_PER_SCENE,
        )

        # ── Writer: написать или переписать сцену ─────────────────────────────
        try:
            if iteration == 1:
                scene_text, scene_diff = _retry(
                    lambda: writer.write_scene(
                        scene_plan=scene_plan,
                        memory_context=memory_context,
                        approved_scenes=approved_scenes,
                    ),
                    label=f"Writer.write_scene (сцена {scene_plan.get('index', '?') + 1})",
                )
            else:
                # Снимаем snapshot переменных замыкания явно,
                # иначе lambda захватит имя, а не значение
                _text_snap = scene_text
                _feedback_snap = critic_feedback
                _iter_snap = iteration

                scene_text, scene_diff = _retry(
                    lambda: writer.rewrite_scene(
                        original_text=_text_snap,
                        critic_feedback=_feedback_snap,
                        scene_plan=scene_plan,
                        memory_context=memory_context,
                        iteration=_iter_snap,
                    ),
                    label=f"Writer.rewrite_scene (итерация {iteration})",
                )

        except Exception as exc:
            logger.error(
                "❌ Writer не справился после всех retry на итерации %d: %s",
                iteration, exc,
            )
            if scene_text:
                # Есть хоть какой-то текст от предыдущей итерации — принимаем
                logger.warning("⚠️  Принимаем текст предыдущей итерации")
                scene_approved = True
            break

        logger.info(
            "  ✍️  Черновик готов: %d символов, %d слов",
            len(scene_text),
            len(scene_text.split()),
        )

        # ── Critic: проверить черновик ─────────────────────────────────────────
        logger.info("  🔍 Критик проверяет сцену...")

        try:
            is_approved, critic_feedback = _retry(
                lambda: critic.review_scene(
                    scene_text=scene_text,
                    memory_context=memory_context,
                    scene_plan=scene_plan,
                    approved_scenes=approved_scenes,
                ),
                label=f"Critic.review_scene (итерация {iteration})",
            )
        except Exception as exc:
            logger.error(
                "❌ Critic недоступен на итерации %d: %s. "
                "Принимаем черновик автоматически.",
                iteration, exc,
            )
            # Критик недоступен — не блокируем систему, принимаем черновик
            scene_approved = True
            break

        # ── Обрабатываем решение Критика ──────────────────────────────────────
        if is_approved:
            logger.info(
                "  ✅ [APPROVE] Сцена «%s» одобрена на итерации %d",
                scene_plan.get("title", "?"),
                iteration,
            )
            scene_approved = True
            break

        # Критик нашёл замечания
        logger.info(
            "  ❌ Замечания Критика (итерация %d / %d):\n  %s",
            iteration,
            MAX_ITERATIONS_PER_SCENE,
            critic_feedback.replace("\n", "\n  "),
        )

        if iteration < MAX_ITERATIONS_PER_SCENE:
            logger.info(
                "  ↩️  Отправляю на переработку (следующая итерация: %d)...",
                iteration + 1,
            )
        else:
            logger.warning(
                "  ⚠️  Лимит итераций (%d) исчерпан для сцены «%s». "
                "Принимаем последний вариант.",
                MAX_ITERATIONS_PER_SCENE,
                scene_plan.get("title", "?"),
            )
            # После исчерпания лимита принимаем лучший доступный вариант
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
    - Дмитрий Воронов (племянник барона, 26 лет, игрок)
    - Мадам Соле (личная секретарша, 45 лет, знает все тайны)
    - Афанасий (дворецкий, 60 лет, служит семье 30 лет)

    Создай запутанный, атмосферный детектив с неожиданным разоблачением. 
    Все сюжетные повороту должны быть связаны и обоснованы, не должно быть сюжетных дыр.
    Читатель должен легко понимать, что происходит в рассказе не додумывая детали. 
    Максимальная длина рассказа 2500 слов.
    """.strip()

    result = run_autonomous_agency(INITIAL_PROMPT)

    if result:
        print(f"\n✅ Готово! Рассказ сохранён: {result}")
    else:
        print("\n❌ Не удалось создать рассказ")
        sys.exit(1)