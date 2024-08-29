from __future__ import annotations

import functools
import importlib
import pkgutil
import threading
import logging
from typing import Any, Callable, Optional, Sequence

import tiktoken_ext
from tiktoken.core import Encoding

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_lock = threading.RLock()
ENCODINGS: dict[str, Encoding] = {}
ENCODING_CONSTRUCTORS: Optional[dict[str, Callable[[], dict[str, Any]]]] = None

@functools.lru_cache()
def _available_plugin_modules() -> Sequence[str]:
    """Return a sequence of available plugin module names."""
    mods = []
    plugin_mods = pkgutil.iter_modules(tiktoken_ext.__path__, tiktoken_ext.__name__ + ".")
    for _, mod_name, _ in plugin_mods:
        mods.append(mod_name)
    logger.debug(f"Available plugin modules: {mods}")
    return mods

def _find_constructors() -> None:
    """Find and register encoding constructors from available plugins."""
    global ENCODING_CONSTRUCTORS
    with _lock:
        if ENCODING_CONSTRUCTORS is not None:
            return
        ENCODING_CONSTRUCTORS = {}

        for mod_name in _available_plugin_modules():
            try:
                mod = importlib.import_module(mod_name)
                constructors = mod.ENCODING_CONSTRUCTORS
                for enc_name, constructor in constructors.items():
                    if enc_name in ENCODING_CONSTRUCTORS:
                        logger.error(f"Duplicate encoding name {enc_name} in tiktoken plugin {mod_name}")
                        raise ValueError(
                            f"Duplicate encoding name {enc_name} in tiktoken plugin {mod_name}"
                        )
                    ENCODING_CONSTRUCTORS[enc_name] = constructor
                    logger.info(f"Registered encoding: {enc_name} from module: {mod_name}")
            except AttributeError as e:
                logger.error(f"tiktoken plugin {mod_name} does not define ENCODING_CONSTRUCTORS")
                raise ValueError(
                    f"tiktoken plugin {mod_name} does not define ENCODING_CONSTRUCTORS"
                ) from e
            except ImportError as e:
                logger.error(f"Error importing module {mod_name}: {e}")
                raise

def get_encoding(encoding_name: str) -> Encoding:
    """Retrieve or create an encoding instance by name."""
    if encoding_name in ENCODINGS:
        return ENCODINGS[encoding_name]

    with _lock:
        if encoding_name in ENCODINGS:
            return ENCODINGS[encoding_name]

        if ENCODING_CONSTRUCTORS is None:
            _find_constructors()
            assert ENCODING_CONSTRUCTORS is not None

        if encoding_name not in ENCODING_CONSTRUCTORS:
            available_plugins = _available_plugin_modules()
            logger.error(f"Unknown encoding {encoding_name}. Plugins found: {available_plugins}")
            raise ValueError(
                f"Unknown encoding {encoding_name}. Plugins found: {available_plugins}"
            )

        constructor = ENCODING_CONSTRUCTORS[encoding_name]
        enc = Encoding(**constructor())
        ENCODINGS[encoding_name] = enc
        logger.info(f"Created and cached encoding: {encoding_name}")
        return enc

def list_encoding_names() -> list[str]:
    """List all available encoding names."""
    with _lock:
        if ENCODING_CONSTRUCTORS is None:
            _find_constructors()
            assert ENCODING_CONSTRUCTORS is not None
        encoding_names = list(ENCODING_CONSTRUCTORS)
        logger.debug(f"Available encoding names: {encoding_names}")
        return encoding_names
