from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

DEFAULT_FORMATTER = logging.Formatter(
    "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def _console_handler(level: int) -> logging.StreamHandler:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(DEFAULT_FORMATTER)
    handler.setLevel(level)
    return handler


def _file_handler(log_dir: Path, filename: str = "system.log") -> RotatingFileHandler:
    log_dir.mkdir(parents=True, exist_ok=True)
    path = str(log_dir / filename)
    handler = RotatingFileHandler(
        filename=path,
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    handler.setFormatter(DEFAULT_FORMATTER)
    handler.setLevel(logging.INFO)
    return handler


def _has_file_handler(logger: logging.Logger) -> bool:
    return any(isinstance(h, RotatingFileHandler) for h in logger.handlers)


def setup_logger(name: str, log_dir: Optional[Path] = None, level: int = logging.INFO) -> logging.Logger:
    
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Add console handler if none exists
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        logger.addHandler(_console_handler(level))

    # Add file handler if requested and not already present
    if log_dir and not _has_file_handler(logger):
        try:
            logger.addHandler(_file_handler(log_dir))
        except Exception:
            logger.exception("Failed to initialize file logging; continuing without file handler")

    # Ensure we don't propagate to root logger (avoid duplicate output)
    logger.propagate = False
    return logger