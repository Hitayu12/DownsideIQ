"""Market-context features (spec §8.5, §9).

Quantifies whether the target's move is company-specific, sector-driven, or
broad-market-driven: rolling beta/correlation to market & sector, relative
strength, regime flags, and an idiosyncratic-move score. All rolling stats are
backward-looking. Missing context assets degrade gracefully (columns become NaN
rather than raising).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _log_returns(df: pd.DataFrame | None) -> pd.Series | None:
    if df is None or df.empty or "close" not in df:
        return None
    c = df["close"].astype(float).sort_index()
    return np.log(c / c.shift(1))


def _first_available(symbols: list[str], context: dict[str, pd.DataFrame]) -> str | None:
    # Context assets are saved with '^' replaced by '_' (e.g. ^VIX -> _VIX).
    for s in symbols:
        for key in (s, s.replace("^", "_")):
            if key in context and not context[key].empty:
                return key
    return None


def compute_market_context_features(
    prices: pd.DataFrame,
    context: dict[str, pd.DataFrame],
    ticker_cfg: dict,
) -> pd.DataFrame:
    """Return market-context features aligned to ``prices.index``."""
    if prices.empty:
        return pd.DataFrame()

    idx = prices.sort_index().index
    tgt = _log_returns(prices)
    out = pd.DataFrame(index=idx)

    market_sym = _first_available(ticker_cfg.get("market_etfs", []), context)
    sector_sym = _first_available(ticker_cfg.get("sector_etfs", []), context)
    vix_sym = _first_available([ticker_cfg.get("vol_proxy", "^VIX")], context)
    peers = ticker_cfg.get("peers", [])

    def aligned(sym: str | None) -> pd.Series | None:
        if sym is None:
            return None
        r = _log_returns(context.get(sym))
        return r.reindex(idx) if r is not None else None

    mkt = aligned(market_sym)
    sec = aligned(sector_sym)

    def rolling_beta(asset: pd.Series | None, window: int = 60) -> pd.Series:
        if asset is None or tgt is None:
            return pd.Series(np.nan, index=idx)
        cov = tgt.rolling(window).cov(asset)
        var = asset.rolling(window).var()
        return cov / var.replace(0, np.nan)

    def rolling_corr(asset: pd.Series | None, window: int = 60) -> pd.Series:
        if asset is None or tgt is None:
            return pd.Series(np.nan, index=idx)
        return tgt.rolling(window).corr(asset)

    # --- Market & sector returns ---
    out["market_return_1b"] = mkt if mkt is not None else np.nan
    out["market_return_5b"] = mkt.rolling(5).sum() if mkt is not None else np.nan
    out["sector_return_1b"] = sec if sec is not None else np.nan
    out["sector_return_5b"] = sec.rolling(5).sum() if sec is not None else np.nan

    # --- Beta & correlation ---
    out["market_beta"] = rolling_beta(mkt)
    out["sector_beta"] = rolling_beta(sec)
    out["correlation_to_market"] = rolling_corr(mkt)
    out["correlation_to_sector"] = rolling_corr(sec)

    # --- Relative strength (target cum return - benchmark cum return, 20b) ---
    if tgt is not None:
        tgt_cum = tgt.rolling(20).sum()
        out["relative_strength_vs_market"] = (
            tgt_cum - mkt.rolling(20).sum() if mkt is not None else np.nan
        )
        out["relative_strength_vs_sector"] = (
            tgt_cum - sec.rolling(20).sum() if sec is not None else np.nan
        )

    # --- Peer group ---
    peer_rets = [aligned(_first_available([p], context)) for p in peers]
    peer_rets = [r for r in peer_rets if r is not None]
    if peer_rets:
        peer_mean = pd.concat(peer_rets, axis=1).mean(axis=1)
        out["peer_group_return_1b"] = peer_mean
        if tgt is not None:
            out["return_vs_peers_1b"] = tgt - peer_mean
    else:
        out["peer_group_return_1b"] = np.nan

    # --- Selloff flags ---
    out["broad_market_selloff_flag"] = (
        (out["market_return_1b"] < -0.02).astype(float) if mkt is not None else np.nan
    )
    out["sector_selloff_flag"] = (
        (out["sector_return_1b"] < -0.02).astype(float) if sec is not None else np.nan
    )

    # --- Correlation spike (current vs trailing median) ---
    cur_corr = out["correlation_to_market"]
    base_corr = cur_corr.rolling(120, min_periods=20).median()
    out["correlation_spike_score"] = (cur_corr - base_corr).clip(lower=0)

    # --- Idiosyncratic (stock-specific) move: residual of target vs beta*market ---
    if tgt is not None and mkt is not None:
        residual = tgt - out["market_beta"] * mkt
        resid_std = residual.rolling(60).std()
        out["stock_specific_move_score"] = (residual.abs() / resid_std.replace(0, np.nan)).clip(0, 3) / 3.0
    else:
        out["stock_specific_move_score"] = np.nan

    # --- VIX level & change ---
    if vix_sym is not None:
        vix_close = context[vix_sym]["close"].astype(float).reindex(idx)
        out["vix_level"] = vix_close
        out["vix_change_1b"] = vix_close.pct_change()
    else:
        out["vix_level"] = np.nan
        out["vix_change_1b"] = np.nan

    # --- Market regime label (risk_on / neutral / risk_off) ---
    out["market_regime"] = _regime_label(out)
    return out


def _regime_label(out: pd.DataFrame) -> pd.Series:
    """Coarse regime: risk_off if market weak / VIX elevated, risk_on if strong."""
    mkt5 = out.get("market_return_5b")
    vix = out.get("vix_level")
    regime = pd.Series("neutral", index=out.index, dtype="object")
    if mkt5 is not None:
        regime = regime.mask(mkt5 < -0.03, "risk_off")
        regime = regime.mask(mkt5 > 0.03, "risk_on")
    if vix is not None:
        regime = regime.mask(vix > 28, "risk_off")
    return regime
