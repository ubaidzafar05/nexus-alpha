"""Structured logging setup for NEXUS-ALPHA."""

from __future__ import annotations

import logging
import sys
from typing import Any

try:
    import structlog
except ImportError:  # pragma: no cover - exercised only in minimal environments
    structlog = None  # type: ignore[assignment]


class StdlibStructuredLogger:
    """Compat logger that accepts structlog-style keyword fields."""

    def __init__(self, name: str):
        self._logger = logging.getLogger(name)

    def _emit(self, level: int, event: str, **fields: Any) -> None:
        if fields:
            extras = " ".join(f"{k}={fields[k]!r}" for k in sorted(fields))
            message = f"{event} {extras}"
        else:
            message = event
        self._logger.log(level, message)

    def debug(self, event: str, **fields: Any) -> None:
        self._emit(logging.DEBUG, event, **fields)

    def info(self, event: str, **fields: Any) -> None:
        self._emit(logging.INFO, event, **fields)

    def warning(self, event: str, **fields: Any) -> None:
        self._emit(logging.WARNING, event, **fields)

    def error(self, event: str, **fields: Any) -> None:
        self._emit(logging.ERROR, event, **fields)

    def exception(self, event: str, **fields: Any) -> None:
        if fields:
            extras = " ".join(f"{k}={fields[k]!r}" for k in sorted(fields))
            message = f"{event} {extras}"
        else:
            message = event
        self._logger.exception(message)


def setup_logging(log_level: str = "INFO") -> None:
    """Configure structured logging with structlog."""
    if structlog is None:
        logging.basicConfig(
            level=getattr(logging, log_level.upper(), logging.INFO),
            format="%(asctime)s %(levelname)s %(name)s %(message)s",
            stream=sys.stderr,
        )
        return

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            structlog.processors.TimeStamper(fmt="iso"),
            (
                structlog.processors.JSONRenderer()
                if log_level != "DEBUG"
                else structlog.dev.ConsoleRenderer()
            ),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, log_level.upper(), logging.INFO)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str):
    """Get a named logger instance."""
    if structlog is None:
        return StdlibStructuredLogger(name)
    return structlog.get_logger(name)
