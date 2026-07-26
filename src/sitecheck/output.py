"""Output contract shared by every command.

Two rules, enforced everywhere:

1. ``--json`` prints a single JSON object to stdout and nothing else.
2. Exit codes are load-bearing: ``0`` success, non-zero failure with one
   human-readable line on stderr.

Commands call :func:`emit` for success payloads and raise :class:`CliError`
for failures. They never ``print`` ad-hoc.
"""

from __future__ import annotations

import json
import math
import sys
from typing import Any

import typer
from rich.console import Console

_console = Console()
_err = Console(stderr=True)

# Exit code carried by a completed audit whose verdict is not trustworthy, so CI can
# gate on provenance the same way it gates on a failing test.
EXIT_VERDICT_FAILED = 2


class CliError(Exception):
    """A failure that maps to a non-zero exit code and one stderr line."""

    def __init__(self, message: str, code: int = 1) -> None:
        super().__init__(message)
        self.message = message
        self.code = code


def _json_safe(value: Any) -> Any:
    """NaN and Infinity are not JSON; carry them as null so parsers don't break."""
    if isinstance(value, float):
        return None if math.isnan(value) or math.isinf(value) else value
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def emit(payload: dict[str, Any], *, json_out: bool, human: str | None = None) -> None:
    """Emit a success payload.

    With ``json_out`` the payload is the entire stdout. Otherwise a friendly
    ``human`` string (or a pretty dump of the payload) is printed.
    """
    if json_out:
        sys.stdout.write(json.dumps(_json_safe(payload), default=str) + "\n")
        return
    if human is not None:
        _console.print(human)
    else:
        _console.print_json(data=_json_safe(payload))


def fail(err: CliError, *, json_out: bool) -> None:
    """Render a failure and exit non-zero.

    In JSON mode the error is still a single JSON object on stdout so an agent
    parsing stdout never has to special-case the error path.
    """
    if json_out:
        sys.stdout.write(json.dumps({"ok": False, "error": err.message}) + "\n")
    else:
        _err.print(f"[red]error:[/red] {err.message}")
    raise typer.Exit(code=err.code)
