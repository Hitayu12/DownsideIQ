"""Back-compat shim — canonical implementation now lives in ``src.core.config``.

Kept so existing MVP modules/tests importing ``src.utils.config_loader`` keep
working during the production migration.
"""
from src.core.config import (  # noqa: F401
    PROJECT_ROOT,
    api_key_status,
    data_dir,
    get_env,
    get_event_playbook,
    get_risk_limits,
    get_settings,
    get_ticker_config,
    get_tickers,
    has_key,
    project_root,
)
