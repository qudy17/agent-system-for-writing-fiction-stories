"""
Клиент для общения с MCP-сервером памяти через stdio (JSON-RPC 2.0).

Оборачивает subprocess-коммуникацию в удобный Python-интерфейс.
Используется orchestrator.py для всех CRUD-операций с онтологической памятью.

Пример использования:
    client = OntologyMemoryMCP(server_path, story_dir)
    client.start()
    ctx = client.get_context("present")
    client.update_state({"present": {"world_state": {"time": "night"}}})
    client.close()
"""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


class OntologyMemoryMCP:
    """
    Клиент для MCP memory_server.py.

    Общается через stdin/stdout JSON-RPC 2.0.
    Использует бинарные потоки + явное UTF-8 кодирование/декодирование
    чтобы избежать проблем с системной кодировкой Windows (cp1251).
    """

    def __init__(self, server_path: Path, story_dir: Path):
        """
        Args:
            server_path : Путь к mcp/memory_server.py
            story_dir   : Директория для хранения файлов памяти
        """
        self.server_path = server_path
        self.story_dir = story_dir
        self._process: Optional[subprocess.Popen[bytes]] = None
        self._stdin: Optional[io.BufferedWriter] = None
        self._stdout: Optional[io.BufferedReader] = None
        self._next_id: int = 1

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Запустить MCP-сервер как дочерний процесс."""
        if self._process is not None:
            return

        env = os.environ.copy()
        env["STORY_DIR"] = str(self.story_dir)
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONLEGACYWINDOWSSTDIO"] = "0"

        self._process = subprocess.Popen(
            [sys.executable, "-X", "utf8", str(self.server_path)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=False,
            env=env,
            cwd=str(self.server_path.parent.parent),
        )

        assert self._process.stdin is not None
        assert self._process.stdout is not None

        self._stdin = self._process.stdin
        self._stdout = self._process.stdout

    def close(self) -> None:
        """Остановить MCP-сервер."""
        if self._process is None:
            return
        try:
            self._process.terminate()
            self._process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._process.kill()
        finally:
            self._process = None
            self._stdin = None
            self._stdout = None

    # ── Низкоуровневая коммуникация ────────────────────────────────────────────

    def _write_line(self, data: str) -> None:
        """
        Записать строку в stdin сервера в UTF-8.

        Args:
            data : Строка для отправки (без символа новой строки)

        Raises:
            RuntimeError : Если процесс не запущен
        """
        if self._stdin is None:
            raise RuntimeError("MCP процесс не запущен")

        raw = (data + "\n").encode("utf-8")
        self._stdin.write(raw)
        self._stdin.flush()

    def _read_line(self) -> str:
        """
        Прочитать строку из stdout сервера, декодировав из UTF-8.

        Returns:
            Декодированная строка без символа новой строки

        Raises:
            RuntimeError : Если процесс не запущен или вернул пустой ответ
        """
        if self._stdout is None:
            raise RuntimeError("MCP процесс не запущен")

        raw = self._stdout.readline()

        if not raw:
            stderr_text = ""
            if self._process and self._process.stderr:
                try:
                    stderr_bytes = self._process.stderr.read(4096)
                    stderr_text = stderr_bytes.decode("utf-8", errors="replace")
                except Exception:
                    pass
            raise RuntimeError(
                f"Нет ответа от MCP memory_server (процесс завершился). "
                f"stderr: {stderr_text[:500]}"
            )

        return raw.decode("utf-8", errors="replace").strip()

    def _call(
        self,
        tool_name: str,
        arguments: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Выполнить вызов инструмента через JSON-RPC 2.0.

        Args:
            tool_name  : Имя инструмента (например "memory_get_context")
            arguments  : Аргументы инструмента

        Returns:
            result-поле из JSON-RPC ответа

        Raises:
            RuntimeError : Если сервер вернул ошибку или нет ответа
        """
        if self._process is None:
            self.start()

        request = {
            "jsonrpc": "2.0",
            "id": self._next_id,
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments or {},
            },
        }
        self._next_id += 1
        request_json = json.dumps(request, ensure_ascii=False)

        # Отправляем через бинарный поток
        self._write_line(request_json)

        # Читаем ответ через бинарный поток
        response_line = self._read_line()

        if not response_line:
            raise RuntimeError(
                f"MCP вернул пустую строку на запрос '{tool_name}'"
            )

        try:
            response = json.loads(response_line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"MCP вернул невалидный JSON: {exc}\n"
                f"Строка: {response_line[:300]}"
            ) from exc

        if "error" in response:
            raise RuntimeError(
                f"MCP ошибка [{tool_name}]: "
                f"{response['error'].get('message', response['error'])}"
            )

        return response["result"]

    # ── Высокоуровневый API (CRUD) ─────────────────────────────────────────────

    def initialize(
        self,
        title: str,
        initial_lore: Dict[str, Any],
        scene_plan: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        CREATE: Инициализировать память начальным лором и планом сцен.

        Args:
            title        : Название истории
            initial_lore : JSON-лор {past: {...}, present: {...}, future: {...}}
            scene_plan   : [{index, title, goal, key_event}, ...]

        Returns:
            Результат инициализации.
        """
        return self._call("memory_initialize", {
            "title": title,
            "initial_lore": initial_lore,
            "scene_plan": scene_plan,
        })

    def get_context(self, time_layer: str = "all") -> Dict[str, Any]:
        """
        READ: Получить контекст по временному слою.

        Args:
            time_layer : "past" | "present" | "future" | "all"

        Returns:
            Контекст запрошенного слоя.
        """
        return self._call("memory_get_context", {"time_layer": time_layer})

    def update_state(self, diff: Dict[str, Any]) -> Dict[str, Any]:
        """
        UPDATE: Применить изменения к состоянию памяти.

        Args:
            diff : Только изменённые поля (не полный слепок!).
                   Пример: {"present": {"clues": [{"id": "knife", ...}]},
                             "advance_scene_index": True}

        Returns:
            Обновлённое мета-состояние.
        """
        return self._call("memory_update", {"diff": diff})

    def get_full_memory(self) -> Dict[str, Any]:
        """
        READ: Получить полный снимок памяти.

        Returns:
            Вся онтологическая память целиком.
        """
        return self._call("memory_get_full", {})

    def add_approved_scene(
        self,
        scene_index: int,
        title: str,
        text: str,
    ) -> Dict[str, Any]:
        """
        CREATE: Сохранить одобренную сцену в архив.

        Args:
            scene_index : Номер сцены (0-based)
            title       : Заголовок сцены
            text        : Полный текст сцены

        Returns:
            Подтверждение сохранения.
        """
        return self._call("memory_add_scene", {
            "scene_index": scene_index,
            "title": title,
            "text": text,
        })

    def get_approved_scenes(self) -> Dict[str, Any]:
        """
        READ: Получить все одобренные сцены.

        Returns:
            {meta, approved_scenes, total}
        """
        return self._call("memory_get_scenes", {})

    def reset(self) -> Dict[str, Any]:
        """
        DELETE: Полностью сбросить память.

        Returns:
            Подтверждение сброса.
        """
        return self._call("memory_reset", {})