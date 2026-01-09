from __future__ import annotations

from pathlib import Path
from typing import Optional

# light-weight top-level imports only
from .config import config_by_name, DevelopmentConfig, ProductionConfig
from .constants import (
    APP_NAME,
    CSV_REQUIRED_COLUMNS,
    CURRENCY_SYMBOL,
    GENDER_OPTIONS,
    ML_CONSTRAINTS,
    REGION_OPTIONS,
    SMOKER_OPTIONS,
    VERSION,
)
from .env_loader import load_environment

__all__ = (
    "APP_NAME",
    "Config",
    "CSV_REQUIRED_COLUMNS",
    "CURRENCY_SYMBOL",
    "DevelopmentConfig",
    "ML_CONSTRAINTS",
    "ProductionConfig",
    "REGION_OPTIONS",
    "SMOKER_OPTIONS",
    "VERSION",
    "config_by_name",
    "init_core",
    "load_environment",
)

# alias for backward compatibility; config.py should provide get_config too
try:
    from .config import Config  # type: ignore
except Exception:
    Config = DevelopmentConfig  # fallback


def init_core(
    env_path: Optional[Path] = None,
    log_dir: Optional[Path] = None,
    logger_name: str = "app",
    log_level: int | None = None,
    override_env: bool = False,
) -> bool:
    """
    Initialize core subsystems: load .env and configure logging.

    Returns True if initialization completed without unhandled exceptions.
    """
    # load environment (thread-safe, per-process)
    try:
        load_environment(env_path=env_path, override=override_env)
    except Exception as exc:
        # if env loading fails, log via basic logging and continue
        import logging
        logging.getLogger(__name__).exception("Failed to load environment: %s", exc)
        # continue to attempt logger setup

    # determine default log level
    if log_level is None:
        try:
            log_level = 10 if getattr(Config, "DEBUG", False) else 20
        except Exception:
            log_level = 20

    # lazy import of setup_logger to reduce circular import risk
    try:
        from .logging import setup_logger  # local import
        setup_logger(logger_name, log_dir=log_dir, level=log_level)
    except Exception as exc:
        import logging
        logging.getLogger(__name__).exception("Failed to setup logger: %s", exc)
        return False

    return True