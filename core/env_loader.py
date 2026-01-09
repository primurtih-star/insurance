from __future__ import annotations

import logging
import os
from pathlib import Path
from threading import Lock
from typing import Optional, Union, Dict, Any

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# module-level state (per-process)
_ENV_LOADED: bool = False
_env_lock = Lock()


def is_env_loaded() -> bool:
    """Return True if environment has been loaded in this process."""
    return _ENV_LOADED


def load_environment(
    env_path: Optional[Union[Path, str]] = None,
    override: bool = False,
    force_reload: bool = False,
    dotenv_kwargs: Optional[Dict[str, Any]] = None,
) -> bool:
    """
    Load environment variables from a .env file in a thread-safe manner.

    Args
      env_path: Path or str to .env file. If None, defaults to project root /.env.
      override: If True, variables from .env override existing os.environ.
      force_reload: If True, reload even if previously loaded in this process.
      dotenv_kwargs: Optional kwargs forwarded to python-dotenv.load_dotenv.

    Returns
      True if a .env file was found and loaded (or already loaded and not force_reload).
      False if no .env file was found or loading failed.
    
    Note
      The loaded flag is per-process. In multi-process WSGI deployments each worker
      must call this function independently.
    """
    global _ENV_LOADED
    dotenv_kwargs = dotenv_kwargs or {}

    with _env_lock:
        if _ENV_LOADED and not force_reload:
            logger.debug("Environment already loaded in this process; skipping.")
            return True

        # resolve default path: project root /.env
        if env_path is None:
            env_path = Path(__file__).resolve().parent.parent / ".env"
        else:
            env_path = Path(env_path)

        try:
            if env_path.is_file():
                loaded = load_dotenv(dotenv_path=str(env_path), override=override, **dotenv_kwargs)
                if loaded:
                    _ENV_LOADED = True
                    logger.info("Environment loaded from %s", env_path)
                    return True
                else:
                    logger.warning("load_dotenv did not load variables from %s", env_path)
                    return False

            # file not present
            logger.info("No .env file found at %s; using system environment variables", env_path)
            return False

        except Exception:
            logger.exception("Failed to load environment from %s", env_path)
            return False


if __name__ == "__main__":
    # minimal test harness
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    result = load_environment()
    print(f"Loaded: {result} | FLASK_ENV: {os.getenv('FLASK_ENV', 'Not Set')}")