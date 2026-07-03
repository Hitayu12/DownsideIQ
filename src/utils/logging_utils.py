"""Back-compat shim — canonical implementation now lives in ``src.core.logging``."""
from src.core.logging import configure_logging, get_logger  # noqa: F401
