"""
MCP stdio-сервер онтологической памяти для детективных рассказов.

Реализует принцип ТРИЗ (системный оператор): хранит состояние мира
по временным слоям — Прошлое, Настоящее, Будущее.

Протокол: newline-delimited JSON-RPC 2.0
Запуск: python mcp/memory_server.py (через subprocess из orchestrator.py)

Инструменты:
    - memory_initialize  : Создать начальное состояние мира
    - memory_get_context : Получить контекст по временному слою
    - memory_update      : Обновить состояние после сцены
    - memory_get_full    : Получить всю память целиком
    - memory_add_scene   : Добавить финальную сцену в историю
    - memory_get_scenes  : Получить список одобренных сцен
    - memory_reset       : Сбросить память (для новой истории)
"""

from __future__ import annotations

import json
import os
import sys
import io
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


# ─────────────────────────── Константы ────────────────────────────────────────

STORY_DIR = Path(os.getenv("STORY_DIR", "stories")).expanduser().resolve()
MEMORY_FILE = STORY_DIR / "ontology_memory.json"

# Временные слои онтологии (ТРИЗ: Прошлое → Настоящее → Будущее)
TIME_LAYERS = ["past", "present", "future"]


# ─────────────────────────── Схема памяти ─────────────────────────────────────

# memory_server.py

def _empty_time_layer() -> Dict[str, Any]:
    """Создать пустой временной слой онтологии."""
    return {
        "world_state": {
            # Разбиваем на подкатегории — каждая фиксируется отдельно
            "weather": None,        # "снег" | "дождь" | "туман" — фиксируется раз и навсегда
            "time_of_day": None,    # "вечер" | "ночь" | "утро"
            "location": None,       # Основное место действия всей истории
            "atmosphere": None,     # "мрачная" | "напряжённая"
            "temperature": None,    # "холодно" | "промозгло"
        },
        "frozen_facts": {},
        "characters": {},
        "clues": [],
        "clue_index": {},
        "events": [],
        "relationships": {},
        "secrets": [],
    }


def _empty_memory() -> Dict[str, Any]:
    """Создать пустую структуру онтологической памяти."""
    return {
        "meta": {
            "story_title": "",
            "genre": "detective",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "current_scene_index": 0,
            "total_scenes_planned": 0,
            "story_finished": False,
        },
        "past": _empty_time_layer(),
        "present": _empty_time_layer(),
        "future": _empty_time_layer(),
        "clue_registry": {},
        "change_log": [],
        "scene_plan": [],
        "approved_scenes": [],
    }

# ─────────────────────────── Хранилище ────────────────────────────────────────

def _load_memory() -> Dict[str, Any]:
    """Загрузить память из файла или создать новую."""
    STORY_DIR.mkdir(parents=True, exist_ok=True)
    if MEMORY_FILE.exists():
        try:
            return json.loads(MEMORY_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    mem = _empty_memory()
    _save_memory(mem)
    return mem


def _save_memory(memory: Dict[str, Any]) -> None:
    """Сохранить память в файл."""
    STORY_DIR.mkdir(parents=True, exist_ok=True)
    memory["meta"]["updated_at"] = datetime.now().isoformat()
    MEMORY_FILE.write_text(
        json.dumps(memory, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# ─────────────────────────── Инструменты ──────────────────────────────────────

def memory_initialize(arguments: Dict[str, Any]) -> Dict[str, Any]:
    title = str(arguments.get("title") or "Детективная история").strip()
    initial_lore: Dict[str, Any] = arguments.get("initial_lore") or {}
    scene_plan: List[Dict[str, Any]] = arguments.get("scene_plan") or []

    mem = _empty_memory()
    mem["meta"]["story_title"] = title
    mem["meta"]["total_scenes_planned"] = len(scene_plan)
    mem["scene_plan"] = scene_plan

    for layer in ["past", "present", "future"]:
        if layer not in initial_lore:
            continue
        layer_data = initial_lore[layer]
        if not isinstance(layer_data, dict):
            print(
                f"[MEMORY] initialize: слой '{layer}' не является dict "
                f"(тип: {type(layer_data).__name__}), пропускаем",
                file=sys.stderr,
            )
            continue

        expected = _empty_time_layer()
        for key in expected:
            if key not in layer_data:
                continue

            value = layer_data[key]

            if key == "world_state":
                if isinstance(value, str):
                    print(
                        f"[MEMORY] initialize: {layer}.world_state — строка вместо dict, "
                        f"конвертирую в {{'description': value}}",
                        file=sys.stderr,
                    )
                    value = {"description": value}
                elif not isinstance(value, dict):
                    print(
                        f"[MEMORY] initialize: {layer}.world_state — "
                        f"неожиданный тип {type(value).__name__}, пропускаем",
                        file=sys.stderr,
                    )
                    continue

                mem[layer]["world_state"].update(
                    {k: v for k, v in value.items() if v is not None}
                )
                continue

            if key == "characters":
                if not isinstance(value, dict):
                    print(
                        f"[MEMORY] initialize: {layer}.characters — "
                        f"тип {type(value).__name__}, пропускаем",
                        file=sys.stderr,
                    )
                    continue
                mem[layer]["characters"] = value
                continue

            if key in ("clues", "events", "secrets", "relationships"):
                if isinstance(value, list):
                    mem[layer][key] = value
                elif isinstance(value, dict):
                    mem[layer][key] = list(value.values())
                else:
                    print(
                        f"[MEMORY] initialize: {layer}.{key} — "
                        f"тип {type(value).__name__}, пропускаем",
                        file=sys.stderr,
                    )
                continue

            mem[layer][key] = value

    _freeze_canonical_facts(mem)

    _save_memory(mem)
    return {
        "initialized": True,
        "title": title,
        "scenes_planned": len(scene_plan),
        "scene_plan": scene_plan,
    }


def _freeze_canonical_facts(mem: Dict[str, Any]) -> None:
    """
    Заморозить все канонические факты сразу после инициализации.

    Вызывается ОДИН РАЗ при старте истории.
    После этого никакой агент не может изменить:
    - роли и родство персонажей
    - погоду, время суток
    - число и имена подозреваемых
    """
    # Замораживаем world_state из present
    present_world = mem["present"].get("world_state", {})
    if isinstance(present_world, dict):
        for key, value in present_world.items():
            if value is not None and key in _FREEZE_ON_SET:
                _freeze_fact(mem, "present", "world_state", key, value)

    # Замораживаем world_state из past
    past_world = mem["past"].get("world_state", {})
    if isinstance(past_world, dict):
        for key, value in past_world.items():
            if value is not None and key in _FREEZE_ON_SET:
                _freeze_fact(mem, "past", "world_state", key, value)

    # ── Замораживаем персонажей (роли, родство, traits) ──────────────────────
    for layer in ["past", "present"]:
        chars = mem[layer].get("characters", {})
        if not isinstance(chars, dict):
            continue

        frozen = mem[layer].setdefault("frozen_facts", {})

        for name, info in chars.items():
            if not isinstance(info, dict):
                continue

            # Замораживаем роль
            role = info.get("role")
            if role:
                fact_key = f"characters.{name}.role"
                if fact_key not in frozen:
                    frozen[fact_key] = {
                        "value": role,
                        "frozen_at_scene": 0,
                    }

            # Замораживаем traits
            traits = info.get("traits")
            if traits and isinstance(traits, list):
                fact_key = f"characters.{name}.traits"
                if fact_key not in frozen:
                    frozen[fact_key] = {
                        "value": traits,
                        "frozen_at_scene": 0,
                    }

            # Замораживаем relationships
            relationships = info.get("relationships")
            if relationships and isinstance(relationships, dict):
                fact_key = f"characters.{name}.relationships"
                if fact_key not in frozen:
                    frozen[fact_key] = {
                        "value": relationships,
                        "frozen_at_scene": 0,
                    }

    print(
        f"[MEMORY] Заморожено канонических фактов: "
        f"present={len(mem['present'].get('frozen_facts', {}))}, "
        f"past={len(mem['past'].get('frozen_facts', {}))}",
        file=sys.stderr,
    )

def memory_get_context(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Получить контекст по временному слою."""
    time_layer = str(arguments.get("time_layer") or "present").strip().lower()
    mem = _load_memory()

    # Формируем компактный реестр улик для передачи агентам
    clue_registry_summary = {
        clue_id: {
            "description": clue.get("description", ""),
            "status": clue.get("status", "hidden"),
            "location": clue.get("location", ""),
            "introduced_in_scene": clue.get("introduced_in_scene", 0),
        }
        for clue_id, clue in mem.get("clue_registry", {}).items()
    }

    base_response = {
        "meta": mem["meta"],
        "scene_plan": mem["scene_plan"],
        "current_scene": _get_current_scene_plan(mem),
        "clue_registry": clue_registry_summary,
        "frozen_facts": {
            layer: mem[layer].get("frozen_facts", {})
            for layer in TIME_LAYERS
        },
    }

    if time_layer == "all":
        return {**base_response, "past": mem["past"], "present": mem["present"], "future": mem["future"]}

    if time_layer not in TIME_LAYERS:
        raise ValueError(f"Неверный time_layer: {time_layer}")

    return {**base_response, time_layer: mem[time_layer]}

def memory_update(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Обновить состояние памяти после написания сцены."""
    diff: Dict[str, Any] = arguments.get("diff") or {}
    mem = _load_memory()
    
    all_conflicts: List[str] = []

    for layer in TIME_LAYERS:
        if layer in diff and isinstance(diff[layer], dict):
            conflicts = _deep_merge(
                mem[layer],
                diff[layer],
                mem=mem,
                layer=layer,
                path="",
            )
            all_conflicts.extend(conflicts)

    if diff.get("advance_scene_index"):
        mem["meta"]["current_scene_index"] += 1

    idx = mem["meta"]["current_scene_index"]
    total = mem["meta"]["total_scenes_planned"]
    if total > 0 and idx >= total:
        mem["meta"]["story_finished"] = True

    _save_memory(mem)
    
    return {
        "updated": True,
        "current_scene_index": mem["meta"]["current_scene_index"],
        "story_finished": mem["meta"]["story_finished"],
        "meta": mem["meta"],
        "conflicts": all_conflicts,
        "clue_registry_size": len(mem.get("clue_registry", {})),
    }

def memory_get_full(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """
    Получить полный снимок памяти.

    Returns:
        Вся структура памяти целиком.
    """
    return _load_memory()


def memory_add_scene(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """
    Добавить одобренную сцену в архив истории.

    Args:
        arguments:
            scene_index (int) : Номер сцены
            title       (str) : Заголовок сцены
            text        (str) : Текст сцены

    Returns:
        Подтверждение добавления.
    """
    scene_index = int(arguments.get("scene_index") or 0)
    title = str(arguments.get("title") or f"Сцена {scene_index + 1}").strip()
    text = str(arguments.get("text") or "").strip()

    if not text:
        raise ValueError("text сцены не может быть пустым")

    mem = _load_memory()
    mem["approved_scenes"].append({
        "index": scene_index,
        "title": title,
        "text": text,
        "approved_at": datetime.now().isoformat(),
    })
    _save_memory(mem)

    return {
        "added": True,
        "scene_index": scene_index,
        "total_approved": len(mem["approved_scenes"]),
    }


def memory_get_scenes(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """
    Получить все одобренные сцены.

    Returns:
        Список одобренных сцен + мета.
    """
    mem = _load_memory()
    return {
        "meta": mem["meta"],
        "approved_scenes": mem["approved_scenes"],
        "total": len(mem["approved_scenes"]),
    }


def memory_reset(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """
    Полностью сбросить память (начать новую историю).

    Returns:
        Подтверждение сброса.
    """
    mem = _empty_memory()
    _save_memory(mem)
    return {"reset": True, "message": "Память очищена. Готово к новой истории."}


# ─────────────────────────── Вспомогательные функции ──────────────────────────
# Поля world_state, которые замораживаются после первой установки

_FREEZE_ON_SET = {"weather", "time_of_day", "location", "temperature"}


def _register_clue(mem: Dict[str, Any], clue: Dict[str, Any]) -> Dict[str, Any]:
    """
    Зарегистрировать улику в глобальном реестре.
    
    Проверяет:
    - Дубликаты по id
    - Противоречие с уже известными фактами о персонажах/местах
    
    Args:
        mem  : Полная память
        clue : {id, description, location, status, found_by, ...}
    
    Returns:
        Улика (возможно обогащённая) или существующая если дубликат.
    """
    clue_id = clue.get("id", "").strip()
    if not clue_id:
        # Генерируем id из описания
        import hashlib
        clue_id = "clue_" + hashlib.md5(
            clue.get("description", "").encode()
        ).hexdigest()[:8]
        clue = {**clue, "id": clue_id}
    
    registry = mem.setdefault("clue_registry", {})
    
    if clue_id in registry:
        # Улика уже существует — обновляем только статус, не описание
        existing = registry[clue_id]
        if clue.get("status") and clue["status"] != existing.get("status"):
            _log_change(mem, f"clue.{clue_id}.status", existing.get("status"), clue["status"])
            existing["status"] = clue["status"]
        if clue.get("found_by"):
            existing["found_by"] = clue["found_by"]
        return existing
    
    # Новая улика — проверяем логическую связность
    conflicts = _check_clue_consistency(mem, clue)
    if conflicts:
        clue = {**clue, "consistency_warnings": conflicts}
    
    # Обогащаем метаданными
    clue = {
        **clue,
        "introduced_in_scene": mem["meta"]["current_scene_index"],
        "status": clue.get("status", "hidden"),
    }
    
    registry[clue_id] = clue
    return clue


def _check_clue_consistency(mem: Dict[str, Any], clue: Dict[str, Any]) -> List[str]:
    """
    Проверить логическую связность улики с текущим состоянием памяти.
    
    Проверяет:
    1. Локация улики существует в мире
    2. Персонаж, который нашёл улику, жив и был там
    3. Описание не противоречит замороженным фактам
    
    Returns:
        Список предупреждений о несоответствиях.
    """
    warnings: List[str] = []
    present = mem.get("present", {})
    characters = present.get("characters", {})
    
    found_by = clue.get("found_by", "")
    clue_location = clue.get("location", "")
    
    # Проверяем: персонаж, нашедший улику, существует и жив
    if found_by and found_by in characters:
        char = characters[found_by]
        if isinstance(char, dict):
            if char.get("status") == "dead":
                warnings.append(
                    f"Улика найдена мёртвым персонажем '{found_by}'"
                )
            char_location = char.get("location", "")
            if char_location and clue_location:
                if char_location.lower() != clue_location.lower():
                    warnings.append(
                        f"'{found_by}' находится в '{char_location}', "
                        f"но улика найдена в '{clue_location}'"
                    )
    
    return warnings

def _freeze_fact(mem: Dict[str, Any], layer: str, category: str, key: str, value: Any) -> None:
    """
    Зафиксировать факт как неизменяемый.
    
    После вызова любая попытка изменить этот факт будет отклонена
    и залогирована как конфликт.
    
    Args:
        mem      : Полная память
        layer    : "past" | "present" | "future"  
        category : "world_state" | "characters"
        key      : Ключ факта ("weather", "time_of_day", ...)
        value    : Значение для заморозки
    """
    frozen = mem[layer].setdefault("frozen_facts", {})
    fact_key = f"{category}.{key}"
    if fact_key not in frozen:
        frozen[fact_key] = {
            "value": value,
            "frozen_at_scene": mem["meta"]["current_scene_index"],
        }


def _check_frozen_conflict(
    mem: Dict[str, Any],
    layer: str, 
    category: str,
    key: str,
    new_value: Any,
) -> Optional[str]:
    """
    Проверить, не противоречит ли новое значение замороженному факту.
    
    Returns:
        Сообщение о конфликте или None если всё ок.
    """
    frozen = mem[layer].get("frozen_facts", {})
    fact_key = f"{category}.{key}"
    
    if fact_key in frozen:
        frozen_value = frozen[fact_key]["value"]
        frozen_at = frozen[fact_key]["frozen_at_scene"]
        
        # Нормализуем для сравнения
        if str(frozen_value).lower() != str(new_value).lower():
            return (
                f"КОНФЛИКТ: {layer}.{category}.{key} = '{frozen_value}' "
                f"(зафиксировано в сцене {frozen_at + 1}), "
                f"попытка изменить на '{new_value}' ОТКЛОНЕНА"
            )
    return None


def _log_change(
    mem: Dict[str, Any],
    field: str,
    old_value: Any,
    new_value: Any,
) -> None:
    """Записать изменение в лог для отладки."""
    mem["change_log"].append({
        "scene": mem["meta"]["current_scene_index"],
        "field": field,
        "old": old_value,
        "new": new_value,
        "timestamp": datetime.now().isoformat(),
    })
    # Держим только последние 50 записей
    if len(mem["change_log"]) > 50:
        mem["change_log"] = mem["change_log"][-50:]


def _get_current_scene_plan(mem: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Вернуть план текущей сцены по индексу."""
    idx = mem["meta"]["current_scene_index"]
    plan = mem.get("scene_plan") or []
    if 0 <= idx < len(plan):
        return plan[idx]
    return None


def _deep_merge(
    base: Dict[str, Any],
    patch: Dict[str, Any],
    mem: Optional[Dict[str, Any]] = None,
    layer: str = "",
    path: str = "",
) -> List[str]:
    conflicts: List[str] = []

    if not isinstance(patch, dict):
        print(
            f"[MEMORY] _deep_merge: patch не dict (path='{path}', "
            f"тип={type(patch).__name__}), пропускаем",
            file=sys.stderr,
        )
        return conflicts

    for key, value in patch.items():
        current_path = f"{path}.{key}" if path else key

        if value is None:
            continue

        # ── Нормализация типов ─────────────────────────────────────────────────
        if key in base and isinstance(base[key], dict) and isinstance(value, str):
            print(
                f"[MEMORY] ожидался dict по '{current_path}', пришла строка — конвертирую",
                file=sys.stderr,
            )
            value = {"description": value}

        if key in base and isinstance(base[key], list) and isinstance(value, str):
            print(
                f"[MEMORY] ожидался list по '{current_path}', пришла строка — пропускаем",
                file=sys.stderr,
            )
            continue

        # ── Защита world_state ─────────────────────────────────────────────────
        if path == "world_state" and mem and layer:
            conflict = _check_frozen_conflict(mem, layer, "world_state", key, value)
            if conflict:
                conflicts.append(conflict)
                print(f"[MEMORY] {conflict}", file=sys.stderr)
                continue

            if key in _FREEZE_ON_SET and value is not None:
                old_value = base.get(key)
                if old_value is None:
                    _freeze_fact(mem, layer, "world_state", key, value)
                    _log_change(mem, f"{layer}.world_state.{key}", old_value, value)

        # ── Защита атрибутов персонажей ────────────────────────────────────────
        if mem and layer and path.startswith("characters."):
            char_name = path.split(".")[1] if len(path.split(".")) > 1 else ""
            if char_name and key in ("role", "traits", "relationships"):
                frozen = mem[layer].get("frozen_facts", {})
                fact_key = f"characters.{char_name}.{key}"
                if fact_key in frozen:
                    frozen_value = frozen[fact_key]["value"]
                    # Для строк — прямое сравнение, для списков — проверяем изменение
                    is_conflict = False
                    if isinstance(frozen_value, str) and isinstance(value, str):
                        is_conflict = frozen_value.lower() != value.lower()
                    elif isinstance(frozen_value, list) and isinstance(value, list):
                        is_conflict = set(map(str, frozen_value)) != set(map(str, value))
                    elif frozen_value != value:
                        is_conflict = True

                    if is_conflict:
                        conflict_msg = (
                            f"КОНФЛИКТ: {layer}.{fact_key} = '{frozen_value}' "
                            f"(заморожено), попытка изменить на '{value}' ОТКЛОНЕНА"
                        )
                        conflicts.append(conflict_msg)
                        print(f"[MEMORY] {conflict_msg}", file=sys.stderr)
                        continue

        # ── Рекурсия для вложенных dict ────────────────────────────────────────
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            sub_conflicts = _deep_merge(
                base[key], value, mem=mem, layer=layer, path=current_path
            )
            conflicts.extend(sub_conflicts)

        # ── Merge списков ──────────────────────────────────────────────────────
        elif key in base and isinstance(base[key], list) and isinstance(value, list):
            if key == "clues" and mem is not None:
                for item in value:
                    if not isinstance(item, dict):
                        continue
                    registered = _register_clue(mem, item)
                    existing_ids = {
                        c.get("id") for c in base[key] if isinstance(c, dict)
                    }
                    if registered.get("id") not in existing_ids:
                        base[key].append(deepcopy(registered))
            else:
                existing_keys = {
                    str(item.get("id") or item.get("description") or item)
                    for item in base[key]
                    if isinstance(item, dict)
                }
                for item in value:
                    item_key = (
                        str(item.get("id") or item.get("description") or item)
                        if isinstance(item, dict) else str(item)
                    )
                    if item_key not in existing_keys:
                        base[key].append(deepcopy(item))
                        existing_keys.add(item_key)

        # ── Скаляры ────────────────────────────────────────────────────────────
        else:
            if base.get(key) != value:
                _log_change(
                    mem or {"change_log": [], "meta": {"current_scene_index": 0}},
                    f"{layer}.{current_path}",
                    base.get(key),
                    value,
                )
            base[key] = deepcopy(value)

    return conflicts

# ─────────────────────────── JSON-RPC ─────────────────────────────────────────

TOOL_HANDLERS = {
    "memory_initialize": memory_initialize,
    "memory_get_context": memory_get_context,
    "memory_update": memory_update,
    "memory_get_full": memory_get_full,
    "memory_add_scene": memory_add_scene,
    "memory_get_scenes": memory_get_scenes,
    "memory_reset": memory_reset,
}


def _tools_list() -> Dict[str, Any]:
    """Вернуть список доступных инструментов (tools/list)."""
    return {
        "tools": [
            {"name": name, "description": fn.__doc__ or ""}
            for name, fn in TOOL_HANDLERS.items()
        ]
    }


def _handle_request(request: Dict[str, Any]) -> Dict[str, Any]:
    """Обработать один JSON-RPC 2.0 запрос."""
    request_id = request.get("id")
    try:
        method = request.get("method")
        if method == "tools/list":
            result = _tools_list()
        elif method == "tools/call":
            params = request.get("params") or {}
            name = params.get("name")
            if name not in TOOL_HANDLERS:
                raise ValueError(f"Неизвестный инструмент: {name}")
            result = TOOL_HANDLERS[name](params.get("arguments") or {})
        else:
            raise ValueError(f"Неизвестный метод: {method}")
        return {"jsonrpc": "2.0", "id": request_id, "result": result}
    except Exception as exc:
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": -32000, "message": str(exc)},
        }


def main() -> None:
    """Запустить MCP память-сервер в режиме stdio."""
    stdin  = io.TextIOWrapper(
        sys.stdin.buffer,
        encoding="utf-8",
        errors="replace",
        newline="\n",
    )
    stdout = io.TextIOWrapper(
        sys.stdout.buffer,
        encoding="utf-8",
        errors="replace",
        newline="\n",
        write_through=True,
    )

    STORY_DIR.mkdir(parents=True, exist_ok=True)

    for line in stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
            response = _handle_request(request)
        except Exception as exc:
            response = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32700, "message": str(exc)},
            }
        stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
        stdout.flush()

if __name__ == "__main__":
    main()