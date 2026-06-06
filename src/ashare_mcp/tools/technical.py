"""Technical analysis tools: MACD, RSI, KDJ, Bollinger, moving averages, risk metrics."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, NamedTuple

import numpy as np
import pandas as pd
import ta as talib

from ashare_mcp.baostock_client import ZERO_THRESHOLD, df_to_records, lookback_range

if TYPE_CHECKING:
    from collections.abc import Callable

    from mcp.server.fastmcp import FastMCP

    from ashare_mcp.baostock_client import Baostock, Record

TA_FIELDS = "date,code,open,high,low,close,volume"

_ROUND_DIGITS = 4
_TRADING_DAYS_PER_YEAR = 252
_DEFAULT_RISK_FREE_RATE = 0.03
_RSI_WINDOW = 14
_BB_WINDOW = 20
_BB_STD = 2
_WR_WINDOW = 14
_KDJ_WINDOW = 9
_KDJ_SMOOTH = 3
_CCI_WINDOW = 20
_ATR_WINDOW = 14
_ADX_WINDOW = 14
_MFI_WINDOW = 14
_KDJ_K_WEIGHT = 3
_KDJ_D_WEIGHT = 2
_DEFAULT_MA_PERIODS = [5, 10, 20, 50, 120, 250]
_MIN_LOOKBACK_DAYS = 30
# Two return periods is the absolute minimum for variance / covariance.
_MIN_RETURNS_FOR_RISK = 2

# Baostock adjustflag: '1' backward-adjusted, '2' forward-adjusted, '3' raw.
# Trend/momentum indicators use forward-adjusted (continuous price across splits).
# Volume indicators (OBV/MFI) MUST use raw bars — baostock leaves volume
# unadjusted, so mixing adjusted price with raw volume produces false buy/sell
# signals on split or bonus-issue days.
_ADJUSTFLAG: dict[str, str] = {"adjusted": "2", "unadjusted": "3"}


def _fetch_ohlcv(
    bs: Baostock, code: str, start_date: str, end_date: str,
    *, adjustflag: str = "2",
) -> pd.DataFrame:
    """Fetch daily OHLCV. adjustflag: '1' backward, '2' forward (default), '3' raw."""
    df = bs.query(
        "query_history_k_data_plus",
        code=code, fields=TA_FIELDS,
        start_date=start_date, end_date=end_date,
        frequency="d", adjustflag=adjustflag,
    )
    df["date"] = pd.to_datetime(df["date"])
    return df.dropna(subset=["close"])


def _finalize(df: pd.DataFrame) -> list[Record]:
    """Format dates, round numerics, convert to records. Returns the full input range."""
    result = df.copy()
    result["date"] = result["date"].dt.strftime("%Y-%m-%d")
    numeric_cols = result.select_dtypes(include=[np.number]).columns
    result[numeric_cols] = result[numeric_cols].round(_ROUND_DIGITS)
    return df_to_records(result)


class _Bars(NamedTuple):
    """OHLCV series bundle passed to indicator functions."""

    close: pd.Series[float]
    high: pd.Series[float]
    low: pd.Series[float]
    volume: pd.Series[float]


def _ind_macd(b: _Bars) -> dict[str, pd.Series[float]]:
    """MACD trend indicator."""
    m = talib.trend.MACD(b.close)
    return {"MACD": m.macd(), "MACD_signal": m.macd_signal(), "MACD_hist": m.macd_diff()}


def _ind_rsi(b: _Bars) -> dict[str, pd.Series[float]]:
    """Relative Strength Index."""
    return {"RSI": talib.momentum.RSIIndicator(b.close, window=_RSI_WINDOW).rsi()}


def _ind_boll(b: _Bars) -> dict[str, pd.Series[float]]:
    """Bollinger Bands (aliased as BB)."""
    bb = talib.volatility.BollingerBands(b.close, window=_BB_WINDOW, window_dev=_BB_STD)
    return {
        "BB_upper": bb.bollinger_hband(),
        "BB_middle": bb.bollinger_mavg(),
        "BB_lower": bb.bollinger_lband(),
    }


def _ind_wr(b: _Bars) -> dict[str, pd.Series[float]]:
    """Williams %R."""
    return {"WR": talib.momentum.WilliamsRIndicator(b.high, b.low, b.close, lbp=_WR_WINDOW).williams_r()}


def _ind_stoch(b: _Bars) -> dict[str, pd.Series[float]]:
    """Stochastic Oscillator (K, D)."""
    s = talib.momentum.StochasticOscillator(b.high, b.low, b.close)
    return {"STOCH_k": s.stoch(), "STOCH_d": s.stoch_signal()}


def _ind_kdj(b: _Bars) -> dict[str, pd.Series[float]]:
    """KDJ (CN-style Stochastic with J = 3K - 2D)."""
    s = talib.momentum.StochasticOscillator(
        b.high, b.low, b.close, window=_KDJ_WINDOW, smooth_window=_KDJ_SMOOTH,
    )
    k = s.stoch()
    d = s.stoch_signal()
    return {"KDJ_K": k, "KDJ_D": d, "KDJ_J": _KDJ_K_WEIGHT * k - _KDJ_D_WEIGHT * d}


def _ind_cci(b: _Bars) -> dict[str, pd.Series[float]]:
    """Commodity Channel Index."""
    return {"CCI": talib.trend.CCIIndicator(b.high, b.low, b.close, window=_CCI_WINDOW).cci()}


def _ind_atr(b: _Bars) -> dict[str, pd.Series[float]]:
    """Average True Range."""
    return {
        "ATR": talib.volatility.AverageTrueRange(
            b.high, b.low, b.close, window=_ATR_WINDOW,
        ).average_true_range(),
    }


def _ind_adx(b: _Bars) -> dict[str, pd.Series[float]]:
    """Average Directional Index with +DI / -DI."""
    a = talib.trend.ADXIndicator(b.high, b.low, b.close, window=_ADX_WINDOW)
    return {"ADX": a.adx(), "ADX_pos": a.adx_pos(), "ADX_neg": a.adx_neg()}


def _ind_obv(b: _Bars) -> dict[str, pd.Series[float]]:
    """On-Balance Volume. Needs RAW bars (volume is never split-adjusted by baostock)."""
    return {"OBV": talib.volume.OnBalanceVolumeIndicator(b.close, b.volume).on_balance_volume()}


def _ind_mfi(b: _Bars) -> dict[str, pd.Series[float]]:
    """Money Flow Index. Needs RAW bars (volume is never split-adjusted by baostock)."""
    return {
        "MFI": talib.volume.MFIIndicator(
            b.high, b.low, b.close, b.volume, window=_MFI_WINDOW,
        ).money_flow_index(),
    }


_Flavor = Literal["adjusted", "unadjusted"]

# name -> (bar-flavor, compute_fn). Flavor drives which fetch the indicator
# consumes — volume indicators need raw bars to avoid split-day distortion.
# dict insertion order is preserved, so this is also the default indicator
# order and the output column order. Aliases get their own entry pointing at
# the same fn; when both are present in `requested`, the fn runs twice and
# the second pass overwrites identical columns — sub-ms noise next to the
# baostock round-trip.
_INDICATORS: dict[str, tuple[_Flavor, Callable[[_Bars], dict[str, pd.Series[float]]]]] = {
    "MACD":  ("adjusted",   _ind_macd),
    "RSI":   ("adjusted",   _ind_rsi),
    "BOLL":  ("adjusted",   _ind_boll),
    "BB":    ("adjusted",   _ind_boll),  # alias of BOLL
    "WR":    ("adjusted",   _ind_wr),
    "STOCH": ("adjusted",   _ind_stoch),
    "KDJ":   ("adjusted",   _ind_kdj),
    "CCI":   ("adjusted",   _ind_cci),
    "ATR":   ("adjusted",   _ind_atr),
    "ADX":   ("adjusted",   _ind_adx),
    "OBV":   ("unadjusted", _ind_obv),
    "MFI":   ("unadjusted", _ind_mfi),
}


def _bars_from(df: pd.DataFrame) -> _Bars:
    """Extract the four series indicator functions consume."""
    return _Bars(close=df["close"], high=df["high"], low=df["low"], volume=df["volume"])


def _apply_indicators(
    bs: Baostock, code: str, start_date: str, end_date: str,
    indicator_set: set[str],
) -> pd.DataFrame:
    """Fetch the bar flavors required by the requested indicators, then apply them.

    Volume indicators (OBV/MFI) trigger a second fetch with adjustflag='3'; trend
    indicators reuse the forward-adjusted fetch. Cost: 1 round-trip if all
    indicators share a flavor, 2 if both flavors are needed.
    """
    requested = {k.upper() for k in indicator_set}
    flavors_needed: set[_Flavor] = {
        flavor for name, (flavor, _) in _INDICATORS.items() if name in requested
    } or {"adjusted"}

    dfs: dict[_Flavor, pd.DataFrame] = {
        flavor: _fetch_ohlcv(bs, code, start_date, end_date, adjustflag=_ADJUSTFLAG[flavor])
        for flavor in flavors_needed
    }

    # Adjusted price is what users expect to see in the date/close column;
    # fall back to unadjusted when only volume indicators were requested.
    base = dfs["adjusted"] if "adjusted" in dfs else dfs["unadjusted"]
    result = base[["date", "close"]].copy()

    for name, (flavor, fn) in _INDICATORS.items():
        if name not in requested:
            continue
        for col, series in fn(_bars_from(dfs[flavor])).items():
            result[col] = series
    return result


def _compute_risk_stats(
    sdf: pd.DataFrame, bdf: pd.DataFrame, risk_free_rate: float,
) -> dict[str, object]:
    """Compute risk statistics from independent stock and benchmark close series.

    Each side computes its own returns on its complete close series. Only the
    aligned subset is used for beta / correlation / tracking_error / info_ratio
    — measures that genuinely require date-paired returns.

    Naively inner-merging first then computing benchmark return would synthesize
    spurious cross-period returns at any stock suspension gap (the merged frame
    skips bench days the stock missed, so the next bench pct_change jumps over
    the gap). That distorts beta and correlation; per-series + aligned cov keeps
    each metric honest.

    Args:
        sdf: Stock DataFrame with `date` and `close` columns (any extra
            ignored). Already NaN-dropped.
        bdf: Benchmark DataFrame, same shape.
        risk_free_rate: Annualized risk-free rate (decimal, e.g. 0.03).

    """
    sc = sdf.set_index("date")["close"].sort_index()
    bc = bdf.set_index("date")["close"].sort_index()
    rs = sc.pct_change().dropna()
    rb = bc.pct_change().dropna()
    n_s, n_b = len(rs), len(rb)
    if n_s < _MIN_RETURNS_FOR_RISK or n_b < _MIN_RETURNS_FOR_RISK:
        msg = f"not enough returns to compute risk metrics (stock={n_s}, bench={n_b})"
        raise ValueError(msg)

    sqrt_252 = float(np.sqrt(_TRADING_DAYS_PER_YEAR))
    vol_stock = float(rs.std()) * sqrt_252
    vol_bench = float(rb.std()) * sqrt_252

    total_r_stock = float(sc.iloc[-1]) / float(sc.iloc[0]) - 1
    total_r_bench = float(bc.iloc[-1]) / float(bc.iloc[0]) - 1
    ann_r_stock = (1 + total_r_stock) ** (_TRADING_DAYS_PER_YEAR / n_s) - 1
    ann_r_bench = (1 + total_r_bench) ** (_TRADING_DAYS_PER_YEAR / n_b) - 1

    sharpe_stock = (ann_r_stock - risk_free_rate) / vol_stock if vol_stock > ZERO_THRESHOLD else 0.0
    sharpe_bench = (ann_r_bench - risk_free_rate) / vol_bench if vol_bench > ZERO_THRESHOLD else 0.0

    # Aligned subset for relational metrics (beta/corr/tracking_error/info_ratio).
    aligned = pd.concat([rs.rename("s"), rb.rename("b")], axis=1).dropna()
    n_aligned = len(aligned)
    if n_aligned < _MIN_RETURNS_FOR_RISK:
        msg = f"not enough aligned return pairs for beta/corr (got {n_aligned})"
        raise ValueError(msg)
    rs_a, rb_a = aligned["s"], aligned["b"]
    var_b = float(rb_a.var())
    beta = float(rs_a.cov(rb_a)) / var_b if abs(var_b) > ZERO_THRESHOLD else 0.0
    correlation = float(rs_a.corr(rb_a))

    excess = rs_a - rb_a
    tracking_error = float(excess.std()) * sqrt_252
    info_ratio = (ann_r_stock - ann_r_bench) / tracking_error if tracking_error > ZERO_THRESHOLD else 0.0

    # Drawdown directly off the close series: avoids the cumprod() pitfall
    # where cum[0] = 1 + r[0] (not 1.0), which causes a first-day large drop
    # to be invisible in the rolling max baseline.
    rolling_max = sc.expanding().max()
    drawdown = (sc - rolling_max) / rolling_max

    return {
        "trading_days_stock": n_s,
        "trading_days_benchmark": n_b,
        "aligned_pairs": n_aligned,
        "returns": {
            "stock_annual": ann_r_stock,
            "benchmark_annual": ann_r_bench,
            "excess": ann_r_stock - ann_r_bench,
        },
        "risk": {
            "beta": beta,
            "stock_volatility": vol_stock,
            "benchmark_volatility": vol_bench,
            "max_drawdown": float(drawdown.min()),
            "correlation": correlation,
        },
        "risk_adjusted": {
            "sharpe_stock": sharpe_stock,
            "sharpe_benchmark": sharpe_bench,
            "information_ratio": info_ratio,
            "tracking_error": tracking_error,
        },
    }


def _register_indicators(app: FastMCP, bs: Baostock) -> None:
    def get_technical_indicators(
        code: str,
        start_date: str,
        end_date: str,
        indicators: list[str] | None = None,
    ) -> list[dict[str, object]]:
        """Calculate technical indicators for a stock.

        Trend/momentum indicators use forward-adjusted prices. Volume indicators
        (OBV, MFI) transparently fetch a second pass with raw bars because
        baostock does not split-adjust volume — mixing forward-adjusted price
        with raw volume would distort money-flow on split / bonus-issue days.
        Volume indicators thus cost one extra network round-trip when requested.

        Returns one row per trading day in [start_date, end_date]. Indicators
        may be null for the first N rows (warmup period; N depends on window).

        Args:
            code: Stock code.
            start_date: 'YYYY-MM-DD'.
            end_date: 'YYYY-MM-DD'.
            indicators: List from
                ['MACD','RSI','KDJ','BOLL','WR','STOCH','CCI','ATR','ADX','OBV','MFI'].
                Defaults to all.

        """
        if indicators is None:
            # BB is an alias of BOLL in _INDICATORS — drop it from the default
            # so the default run doesn't compute Bollinger twice.
            indicators = [name for name in _INDICATORS if name != "BB"]
        return _finalize(_apply_indicators(
            bs, code, start_date, end_date, {i.upper() for i in indicators},
        ))

    app.tool()(get_technical_indicators)


def _register_moving_averages(app: FastMCP, bs: Baostock) -> None:
    def get_moving_averages(
        code: str,
        start_date: str,
        end_date: str,
        periods: list[int] | None = None,
    ) -> list[dict[str, object]]:
        """Calculate SMA and EMA for multiple periods over the requested date range.

        Returns one row per trading day in [start_date, end_date]. MA values are null
        for the first (period - 1) rows of each MA_<p> / EMA_<p> column.

        Args:
            code: Stock code.
            start_date: 'YYYY-MM-DD'.
            end_date: 'YYYY-MM-DD'.
            periods: Period list, e.g. [5,10,20,50,120,250]. Defaults to common set.

        """
        df = _fetch_ohlcv(bs, code, start_date, end_date)
        if periods is None:
            periods = list(_DEFAULT_MA_PERIODS)

        result = df[["date", "close"]].copy()
        for p in periods:
            if len(df) >= p:
                result[f"SMA_{p}"] = df["close"].rolling(window=p).mean()
                # adjust=False uses the recursive EMA formula
                # (EMA_t = alpha*x_t + (1-alpha)*EMA_{t-1}), matching ta's
                # internal MACD implementation and TradingView / talib convention.
                result[f"EMA_{p}"] = df["close"].ewm(span=p, adjust=False).mean()

        return _finalize(result)

    app.tool()(get_moving_averages)


def _register_risk_metrics(app: FastMCP, bs: Baostock) -> None:
    def calculate_risk_metrics(
        code: str,
        benchmark_code: str = "sh.000300",
        lookback_days: int = 365,
        risk_free_rate: float = _DEFAULT_RISK_FREE_RATE,
    ) -> dict[str, object]:
        """Calculate risk metrics: beta, Sharpe, max drawdown, volatility, correlation.

        Args:
            code: Stock code.
            benchmark_code: Benchmark index, default 'sh.000300' (CSI 300).
            lookback_days: Calendar days to look back. 365 ≈ 1 year, 730 ≈ 2 years.
                Minimum 30. ~245 calendar days ≈ 1 trading year (CN A-share).
            risk_free_rate: Annualized risk-free rate for Sharpe ratio.
                Default ~3% (approximate CN 10Y bond yield); override for non-CN markets.

        """
        if lookback_days < _MIN_LOOKBACK_DAYS:
            msg = f"lookback_days must be >= {_MIN_LOOKBACK_DAYS}, got {lookback_days}"
            raise ValueError(msg)

        start_date, end_date = lookback_range(lookback_days)

        sdf = _fetch_ohlcv(bs, code, start_date, end_date)
        bdf = _fetch_ohlcv(bs, benchmark_code, start_date, end_date)

        stats = _compute_risk_stats(sdf, bdf, risk_free_rate)
        return {
            "code": code,
            "benchmark": benchmark_code,
            "lookback_days": lookback_days,
            "risk_free_rate": risk_free_rate,
            **stats,
        }

    app.tool()(calculate_risk_metrics)


def register(app: FastMCP, bs: Baostock) -> None:
    """Register technical analysis tools with the MCP app."""
    _register_indicators(app, bs)
    _register_moving_averages(app, bs)
    _register_risk_metrics(app, bs)
