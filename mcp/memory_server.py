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

def _empty_time_layer() -> Dict[str, Any]:
    """Создать пустой временной слой онтологии."""
    return {
        "world_state": {},        # Состояние мира: погода, место, время суток и т.д.
        "characters": {},         # Персонажи: {name: {role, status, location, traits}}
        "clues": [],              # Улики: [{id, description, found_by, location, status}]
        "events": [],             # События: [{description, timestamp, participants}]
        "relationships": {},      # Отношения между персонажами: {name: {other: relation}}
        "secrets": [],            # Тайны/скрытые факты (известны только системе)
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
        # План сцен: [{index, title, goal, key_event}]
        "scene_plan": [],
        # Одобренные сцены: [{index, title, text, approved_at}]
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
    """
    Инициализировать память начальным состоянием мира.

    Args:
        arguments:
            title        (str)  : Название истории
            initial_lore (dict) : Начальный JSON-лор от Писателя
            scene_plan   (list) : Список сцен [{index, title, goal, key_event}]

    Returns:
        Подтверждение инициализации с мета-данными.
    """
    title = str(arguments.get("title") or "Детективная история").strip()
    initial_lore: Dict[str, Any] = arguments.get("initial_lore") or {}
    scene_plan: List[Dict[str, Any]] = arguments.get("scene_plan") or []

    mem = _empty_memory()
    mem["meta"]["story_title"] = title
    mem["meta"]["total_scenes_planned"] = len(scene_plan)
    mem["scene_plan"] = scene_plan

    # Заполняем начальный слой "Прошлое" (backstory) и "Настоящее" (starting state)
    for layer in ["past", "present", "future"]:
        if layer in initial_lore:
            layer_data = initial_lore[layer]
            if isinstance(layer_data, dict):
                for key in _empty_time_layer():
                    if key in layer_data:
                        mem[layer][key] = layer_data[key]

    _save_memory(mem)
    return {
        "initialized": True,
        "title": title,
        "scenes_planned": len(scene_plan),
        "scene_plan": scene_plan,
    }


def memory_get_context(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """
    Получить контекст по временному слою.

    Args:
        arguments:
            time_layer (str): "past" | "present" | "future" | "all"

    Returns:
        Данные запрошенного временного слоя + мета-информация.
    """
    time_layer = str(arguments.get("time_layer") or "present").strip().lower()
    mem = _load_memory()

    if time_layer == "all":
        return {
            "meta": mem["meta"],
            "past": mem["past"],
            "present": mem["present"],
            "future": mem["future"],
            "scene_plan": mem["scene_plan"],
            "current_scene": _get_current_scene_plan(mem),
        }

    if time_layer not in TIME_LAYERS:
        raise ValueError(f"Неверный time_layer: {time_layer}. Допустимые: {TIME_LAYERS + ['all']}")

    return {
        "meta": mem["meta"],
        time_layer: mem[time_layer],
        "scene_plan": mem["scene_plan"],
        "current_scene": _get_current_scene_plan(mem),
    }


def memory_update(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """
    Обновить состояние памяти после написания сцены.

    Принимает diff-JSON от Писателя — только изменения, не полный слепок.
    Применяет патч поверх существующего состояния (merge, не replace).

    Args:
        arguments:
            diff (dict): {
                "past":    {...изменения...},
                "present": {...изменения...},
                "future":  {...изменения...},
                "advance_scene_index": bool  — перейти к следующей сцене
            }

    Returns:
        Обновлённое мета-состояние.
    """
    diff: Dict[str, Any] = arguments.get("diff") or {}
    mem = _load_memory()

    for layer in TIME_LAYERS:
        if layer in diff and isinstance(diff[layer], dict):
            _deep_merge(mem[layer], diff[layer])

    # Сдвигаем указатель сцены, если Писатель попросил
    if diff.get("advance_scene_index"):
        mem["meta"]["current_scene_index"] += 1

    # Проверяем, закончена ли история
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
    }


def memory_get_full(arguments: Dict[str, Any]) -> Dict[str, Any]:  # noqa: ARG001
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


def memory_get_scenes(arguments: Dict[str, Any]) -> Dict[str, Any]:  # noqa: ARG001
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


def memory_reset(arguments: Dict[str, Any]) -> Dict[str, Any]:  # noqa: ARG001
    """
    Полностью сбросить память (начать новую историю).

    Returns:
        Подтверждение сброса.
    """
    mem = _empty_memory()
    _save_memory(mem)
    return {"reset": True, "message": "Память очищена. Готово к новой истории."}


# ─────────────────────────── Вспомогательные функции ──────────────────────────

def _get_current_scene_plan(mem: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Вернуть план текущей сцены по индексу."""
    idx = mem["meta"]["current_scene_index"]
    plan = mem.get("scene_plan") or []
    if 0 <= idx < len(plan):
        return plan[idx]
    return None


def _deep_merge(base: Dict[str, Any], patch: Dict[str, Any]) -> None:
    """
    Рекурсивный merge патча поверх базового словаря.

    Правила:
        - dict + dict  → рекурсивный merge
        - list  + list → extend (добавляем новые элементы)
        - любое + scalar → replace
    """
    for key, value in patch.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        elif key in base and isinstance(base[key], list) and isinstance(value, list):
            # Для списков улик/событий — добавляем новые, не дублируем
            existing_descs = {
                str(item.get("id") or item.get("description") or item)
                for item in base[key]
                if isinstance(item, dict)
            }
            for item in value:
                item_key = str(
                    item.get("id") or item.get("description") or item
                ) if isinstance(item, dict) else str(item)
                if item_key not in existing_descs:
                    base[key].append(deepcopy(item))
                    existing_descs.add(item_key)
        else:
            base[key] = deepcopy(value)


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
    STORY_DIR.mkdir(parents=True, exist_ok=True)
    for line in sys.stdin:
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
        sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()