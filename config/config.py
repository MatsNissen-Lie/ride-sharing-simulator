"""Project-wide logging configuration and helpers."""

import logging
from enum import Enum
from pathlib import Path
from typing import Iterable


BASE_DIR = Path(__file__).resolve().parent.parent
"""Absolute project root used for filesystem operations."""


class LoggingScope(str, Enum):
    """Controls whether loggers manage their own handlers or defer to the root."""

    MODULE_ONLY = "MODULE_ONLY"
    GLOBAL = "GLOBAL"


LOGGING_LEVEL = logging.INFO
"""Default level applied to module loggers when scoped locally."""

LOGGING_GLOBAL_LEVEL = logging.INFO
"""Level applied to the root logger when global scoping is enabled."""

LOGGING_SCOPE = LoggingScope.MODULE_ONLY
"""Select between per-module handlers or global propagation."""

_CONSOLE_FORMAT = "[%(levelname)s] %(name)-28s | %(message)s"
_BLOCK_INDENT_WIDTH = len("[INFO] ") + 28 + len(" | ")

_ROOT_HANDLER_ATTR = "_ride_logging_root_handler"
_MODULE_HANDLER_ATTR = "_ride_logging_module_handler"

_formatter = logging.Formatter(_CONSOLE_FORMAT)


def _ensure_single_stream_handler(logger: logging.Logger, attr: str) -> logging.Handler:
    """Attach (or return) a singleton StreamHandler tagged with *attr*."""

    for handler in logger.handlers:
        if getattr(handler, attr, False):
            return handler

    # Remove existing StreamHandlers to avoid duplicate console output.
    for handler in list(logger.handlers):
        if isinstance(handler, logging.StreamHandler) and not getattr(handler, attr, False):
            logger.removeHandler(handler)

    handler = logging.StreamHandler()
    handler.setFormatter(_formatter)
    handler.setLevel(logging.NOTSET)
    setattr(handler, attr, True)
    logger.addHandler(handler)
    return handler


def _remove_module_handlers(logger: logging.Logger) -> None:
    """Detach console handlers previously attached by :func:`get_logger`."""

    for handler in list(logger.handlers):
        if getattr(handler, _MODULE_HANDLER_ATTR, False):
            logger.removeHandler(handler)


def _configure_root_logger() -> logging.Logger:
    """Ensure the root logger is ready with a single console handler."""

    root = logging.getLogger()
    _ensure_single_stream_handler(root, _ROOT_HANDLER_ATTR)
    return root


def _apply_scope(root: logging.Logger, logger: logging.Logger) -> None:
    """Configure *logger* according to the selected :data:`LOGGING_SCOPE`."""

    if LOGGING_SCOPE is LoggingScope.GLOBAL:
        root.setLevel(LOGGING_GLOBAL_LEVEL)
        if logger is root:
            return
        _remove_module_handlers(logger)
        logger.setLevel(logging.NOTSET)
        logger.propagate = True
    else:
        # Keep the root quiet while module loggers handle their own output.
        root.setLevel(logging.WARNING)
        if logger is root:
            return
        handler = _ensure_single_stream_handler(logger, _MODULE_HANDLER_ATTR)
        handler.setLevel(logging.NOTSET)
        logger.setLevel(LOGGING_LEVEL)
        logger.propagate = False


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a configured :class:`logging.Logger`.

    The first call also ensures the root logger is set up with a single
    ``StreamHandler`` using the shared formatter. Subsequent calls reuse the
    existing configuration while updating scope-related settings.
    """

    root = _configure_root_logger()
    target = logging.getLogger(name)
    _apply_scope(root, target)
    return target


def format_log_block(message: str) -> str:
    """Indent multiline messages so continuation lines align under the prefix."""

    if not message:
        return message

    lines: Iterable[str] = message.splitlines()
    iterator = iter(lines)
    first_line = next(iterator, "")
    padding = " " * _BLOCK_INDENT_WIDTH
    remainder = [f"{padding}{line}" if line else padding.rstrip() for line in iterator]
    return "\n".join([first_line, *remainder]) if remainder else first_line


__all__ = [
    "BASE_DIR",
    "LOGGING_LEVEL",
    "LOGGING_GLOBAL_LEVEL",
    "LOGGING_SCOPE",
    "LoggingScope",
    "format_log_block",
    "get_logger",
]
