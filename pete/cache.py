import os
from pathlib import Path

if _xdg_cache := os.getenv("XDG_CACHE_HOME"):
    CACHE_DIR = Path(_xdg_cache) / "pete"
else:
    CACHE_DIR = Path.home() / ".cache" / "pete"
