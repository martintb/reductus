
import copy

from .core import load_instrument, lookup_instrument
from .cache import get_cache
from . import fetch
from configurations import default

DEFAULT_CONFIG = copy.deepcopy(default.config)

def load_update(name="config_overrides"):
    """
    Load named configuration from configurations folder, and update
    a copy of default configuration with user settings
    """
    import importlib
    config_module = importlib.import_module("configurations.{name}".format(name=name))
    config_overrides = copy.deepcopy(config_module.config)
    config = copy.deepcopy(DEFAULT_CONFIG)
    config.update(config_overrides)
    return config

def load_config(name="config", fallback=True):
    """
    Look for configurations defined in the configurations directory
    if the name is not found, use "default_config" if fallback==True
    """
    import importlib

    try:
        config_module = importlib.import_module("configurations.{name}".format(name=name))
        return copy.deepcopy(config_module.config)
    except ImportError:
        if fallback:
            return DEFAULT_CONFIG
        else:
            raise

def apply_config(user_config=None, user_overrides=None):
    if user_config is not None:
        config = copy.deepcopy(user_config)
    else:
        config = copy.deepcopy(DEFAULT_CONFIG)
    
    if user_overrides is not None:
        config.update(user_overrides)

    fetch.DATA_SOURCES = config.get("data_sources", [])
    fetch.FILE_HELPERS = dict([(source["name"], source.get("file_helper_url", None)) for source in fetch.DATA_SOURCES])

    cache_config = config.get('cache', False)
    if cache_config:
        cache_engine = cache_config.get("engine", None)
        cache_params = cache_config.get("params", {})
        cache_compression = cache_config.get("compression", False)
        cache_manager = get_cache()
        if cache_engine == "diskcache":
            cache_manager.use_diskcache(**cache_params)
        elif cache_engine == "redis":
            cache_manager.use_redis(**cache_params)
        else:
            cache_manager.use_memory()

        cache_manager._use_compression = cache_compression

    # Load refl instrument if nothing specified in config.
    # Note: instrument names do not match instrument ids.
    instruments = config.get('instruments', ['refl'])
    for name in instruments:
        load_instrument(name)