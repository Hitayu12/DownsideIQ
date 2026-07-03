"""Back-compat shim — canonical implementation now lives in ``src.core.time``."""
from src.core.time import (  # noqa: F401
    UTC,
    LookAheadError,
    assert_no_future_data,
    ensure_tz_aware,
    filter_strictly_before,
    now_market,
    now_utc,
    to_utc,
)

# Some MVP modules referenced this module-level constant.
MARKET_TZ = None  # resolved lazily; use src.core.time.market_tz() for the live value
