"""Technical analysis tools: MACD, RSI, KDJ, Bollinger, moving averages, risk metrics."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, NamedTuple

import numpy as np
import pandas as pd
import ta as talib

from ashare_mcp.utils import ZERO_THRESHOLD, df_to_records, lookback_range

if TYPE_CHECKING:
    from collections.abc import Callable

    from mcp.server.fastmcp import FastMCP

    from ashare_mcp.baostock_client import Baostock
    from ashare_mcp.utils import Record

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
# Floor on real trading-day bars (not calendar days): 30 observations is the
# central-limit-theorem minimum below which variance / covariance — and the
# beta / Sharpe / volatility built on them — are unreliable. Enforced AFTER the
# fetch on actual bars, because a calendar-day span can't guarantee a bar count
# (a window crossing Spring Festival yields far fewer than a quiet stretch).
_MIN_TRADING_DAYS = 30
# Two return periods is the absolute minimum for variance / covariance.
_MIN_RETURNS_FOR_RISK = 2

# Baostock adjustflag: '1' backward-adjusted, '2' forward-adjusted, '3' raw.
# Trend/momentum indicators use forward-adjusted (continuous price across splits).
# Volume indicators (OBV/MFI) MUST use raw bars — baostock leaves volume
# unadjusted, so mixing adjusted price with raw volume produces false buy/sell
# signals on split or bonus-issue days.
_ADJUSTFLAG: dict[str, str] = {"adjusted": "2", "unadjusted": "3"}


def _fetch_ohlcv(
    bs: Baostock,
    code: str,
    start_date: str,
    end_date: str,
    *,
    adjustflag: str = "2",
) -> pd.DataFrame:
    """Fetch daily OHLCV. adjustflag: '1' backward, '2' forward (default), '3' raw."""
    df = bs.query(
        "query_history_k_data_plus",
        code=code,
        fields=TA_FIELDS,
        start_date=start_date,
        end_date=end_date,
        frequency="d",
        adjustflag=adjustflag,
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
    """KDJ, China convention: K/D are RSV smoothed twice, J = 3K - 2D.

    ta's StochasticOscillator returns an unsmoothed raw RSV as %K, which does not
    match domestic CN terminals. The CN KDJ smooths RSV with a 1/3 recursive
    average (the SMA(X, 3, 1) rule): K = 1/3*RSV + 2/3*K_prev, then D likewise over
    K. An alpha=1/3 EMA is exactly that recursion. STOCH still exposes ta's raw
    American stochastic; only KDJ follows the CN convention here.
    """
    low_min = b.low.rolling(_KDJ_WINDOW).min()
    high_max = b.high.rolling(_KDJ_WINDOW).max()
    rsv = (b.close - low_min) / (high_max - low_min) * 100
    k = rsv.ewm(alpha=1 / _KDJ_SMOOTH, adjust=False).mean()
    d = k.ewm(alpha=1 / _KDJ_SMOOTH, adjust=False).mean()
    return {"KDJ_K": k, "KDJ_D": d, "KDJ_J": _KDJ_K_WEIGHT * k - _KDJ_D_WEIGHT * d}


def _ind_cci(b: _Bars) -> dict[str, pd.Series[float]]:
    """Commodity Channel Index."""
    return {"CCI": talib.trend.CCIIndicator(b.high, b.low, b.close, window=_CCI_WINDOW).cci()}


def _ind_atr(b: _Bars) -> dict[str, pd.Series[float]]:
    """Average True Range."""
    atr = talib.volatility.AverageTrueRange(
        b.high,
        b.low,
        b.close,
        window=_ATR_WINDOW,
    ).average_true_range()
    # ta library fills the warmup span with 0.0 regardless of fillna=False — reads
    # as "zero volatility" instead of "not yet computed", violating this tool's
    # null-warmup contract. Mask the first (window - 1) bars back to NaN. The
    # magic number is pinned by ta's current implementation; an upgrade that
    # shifts the warmup boundary needs a test failure here, not silent drift.
    atr.iloc[: _ATR_WINDOW - 1] = np.nan
    return {"ATR": atr}


def _ind_adx(b: _Bars) -> dict[str, pd.Series[float]]:
    """Average Directional Index with +DI / -DI."""
    a = talib.trend.ADXIndicator(b.high, b.low, b.close, window=_ADX_WINDOW)
    adx, pos, neg = a.adx(), a.adx_pos(), a.adx_neg()
    # Same ta-library 0-fill quirk as _ind_atr. Three distinct warmup lengths:
    # +DI/-DI need (window + 1) bars; ADX itself smooths `window` directional
    # readings on top of that, so its first valid value is at index 2*window - 1.
    pos.iloc[: _ADX_WINDOW + 1] = np.nan
    neg.iloc[: _ADX_WINDOW + 1] = np.nan
    adx.iloc[: 2 * _ADX_WINDOW - 1] = np.nan
    return {"ADX": adx, "ADX_pos": pos, "ADX_neg": neg}


def _ind_obv(b: _Bars) -> dict[str, pd.Series[float]]:
    """On-Balance Volume. Needs RAW bars (volume is never split-adjusted by baostock)."""
    return {"OBV": talib.volume.OnBalanceVolumeIndicator(b.close, b.volume).on_balance_volume()}


def _ind_mfi(b: _Bars) -> dict[str, pd.Series[float]]:
    """Money Flow Index. Needs RAW bars (volume is never split-adjusted by baostock)."""
    return {
        "MFI": talib.volume.MFIIndicator(
            b.high,
            b.low,
            b.close,
            b.volume,
            window=_MFI_WINDOW,
        ).money_flow_index(),
    }


_Flavor = Literal["adjusted", "unadjusted"]

# name -> (bar-flavor, compute_fn, output_columns, warmup_bars). Flavor drives
# which fetch the indicator consumes — volume indicators need raw bars to
# avoid split-day distortion. output_columns lists the keys the compute_fn
# returns; on a short-window IndexError from ta, _safe_compute uses it to
# emit an all-null stub so other indicators in the same call still produce
# values. warmup_bars is the first-non-null offset under default parameters;
# _apply_indicators extends the fetch by 2x this value so the user's
# requested window has no warmup nulls and IIR-based EMAs (MACD/ATR/ADX) have
# room to converge past their first non-null bar.
# dict insertion order is preserved, so this is also the default indicator
# order and the output column order. Aliases get their own entry pointing at
# the same fn; when both are present in `requested`, the fn runs twice and
# the second pass overwrites identical columns — sub-ms noise next to the
# baostock round-trip.
_INDICATORS: dict[
    str,
    tuple[_Flavor, Callable[[_Bars], dict[str, pd.Series[float]]], tuple[str, ...], int],
] = {
    "MACD": ("adjusted", _ind_macd, ("MACD", "MACD_signal", "MACD_hist"), 35),  # slow(26) + signal(9)
    "RSI": ("adjusted", _ind_rsi, ("RSI",), _RSI_WINDOW),
    "BOLL": ("adjusted", _ind_boll, ("BB_upper", "BB_middle", "BB_lower"), _BB_WINDOW),
    "BB": ("adjusted", _ind_boll, ("BB_upper", "BB_middle", "BB_lower"), _BB_WINDOW),  # alias of BOLL
    "WR": ("adjusted", _ind_wr, ("WR",), _WR_WINDOW),
    "STOCH": ("adjusted", _ind_stoch, ("STOCH_k", "STOCH_d"), 16),  # k(14) + smooth(3) - 1
    "KDJ": ("adjusted", _ind_kdj, ("KDJ_K", "KDJ_D", "KDJ_J"), _KDJ_WINDOW),
    "CCI": ("adjusted", _ind_cci, ("CCI",), _CCI_WINDOW),
    "ATR": ("adjusted", _ind_atr, ("ATR",), _ATR_WINDOW),
    "ADX": ("adjusted", _ind_adx, ("ADX", "ADX_pos", "ADX_neg"), 2 * _ADX_WINDOW - 1),
    "OBV": ("unadjusted", _ind_obv, ("OBV",), 0),  # cumulative, no warmup
    "MFI": ("unadjusted", _ind_mfi, ("MFI",), _MFI_WINDOW),
}


def _bars_from(df: pd.DataFrame) -> _Bars:
    """Extract the four series indicator functions consume."""
    return _Bars(close=df["close"], high=df["high"], low=df["low"], volume=df["volume"])


def _safe_compute(
    fn: Callable[[_Bars], dict[str, pd.Series[float]]],
    bars: _Bars,
    columns: tuple[str, ...],
) -> dict[str, pd.Series[float]]:
    """Run an indicator; on short-window failure, return all-null columns.

    `ta` raises on shortage with two different shapes depending on the indicator:
      - IndexError when it writes past the end of a pre-sized array (e.g. ATR's
        window=14 with 4 bars writes atr[13] into a length-4 array).
      - ValueError("negative dimensions are not allowed") when it allocates a
        shape of `n - window` and that goes negative (e.g. ADX with 4 bars).
    Both mean "fewer bars than the window can absorb"; map them to a structured
    null result keyed on `columns` so the rest of the requested indicators in
    the same call still produce values, instead of the whole tool aborting with
    a ta-internal traceback. Pure-pandas indicators (KDJ) don't raise on shortage
    — they emit NaN naturally — and pass through fn(bars).
    """
    try:
        return fn(bars)
    except (IndexError, ValueError):
        return {col: pd.Series(np.nan, index=bars.close.index, dtype=float) for col in columns}


def _warmup_bars_for(requested: set[str]) -> int:
    """Max warmup the requested indicators need, doubled for IIR convergence.

    Each indicator declares its first-non-null offset (`warmup_bars` in
    `_INDICATORS`). EMA-based indicators (MACD/ATR/ADX) keep converging past
    that point, so we double to give early values inside the user's window
    room to settle. OBV-only requests yield 0 (cumulative, no warmup needed).
    """
    return 2 * max(
        (w for name, (_, _, _, w) in _INDICATORS.items() if name in requested),
        default=0,
    )


def _expand_start(start_date: str, warmup_bars: int) -> str:
    """Pull start_date back enough calendar days to cover `warmup_bars` trading days.

    252 trading days ≈ 365 calendar; +15 matches the holiday-cluster buffer
    used elsewhere in this codebase (market._LATEST_TRADE_LOOKBACK_DAYS,
    valuation._*_LOOKBACK_DAYS, derivations._PRICE_LOOKBACK_DAYS) — absorbs
    Spring Festival / National Day clusters without per-holiday math. Used to
    silently extend the baostock fetch before computing indicators so the
    returned [start_date, end_date] range is fully warmed up.
    """
    if warmup_bars <= 0:
        return start_date
    calendar_days = warmup_bars * 365 // _TRADING_DAYS_PER_YEAR + 15
    return lookback_range(calendar_days, end=start_date)[0]


def _apply_indicators(
    bs: Baostock,
    code: str,
    start_date: str,
    end_date: str,
    indicator_set: set[str],
) -> pd.DataFrame:
    """Fetch the bar flavors required by the requested indicators, then apply them.

    Volume indicators (OBV/MFI) trigger a second fetch with adjustflag='3'; trend
    indicators reuse the forward-adjusted fetch. Cost: 1 round-trip if all
    indicators share a flavor, 2 if both flavors are needed.

    Fetches a warmup buffer before `start_date` so indicators are pre-converged
    when they enter the user's window; the returned DataFrame is trimmed back
    to `[start_date, end_date]`.
    """
    requested = {k.upper() for k in indicator_set}
    flavors_needed: set[_Flavor] = {flavor for name, (flavor, _, _, _) in _INDICATORS.items() if name in requested} or {
        "adjusted",
    }

    fetch_start = _expand_start(start_date, _warmup_bars_for(requested))
    dfs: dict[_Flavor, pd.DataFrame] = {
        flavor: _fetch_ohlcv(bs, code, fetch_start, end_date, adjustflag=_ADJUSTFLAG[flavor])
        for flavor in flavors_needed
    }

    # Adjusted price is what users expect to see in the date/close column;
    # fall back to unadjusted when only volume indicators were requested.
    base = dfs["adjusted"] if "adjusted" in dfs else dfs["unadjusted"]
    result = base[["date", "close"]].copy()

    for name, (flavor, fn, cols, _) in _INDICATORS.items():
        if name not in requested:
            continue
        for col, series in _safe_compute(fn, _bars_from(dfs[flavor]), cols).items():
            result[col] = series

    user_start = pd.to_datetime(start_date)
    return result[result["date"] >= user_start].reset_index(drop=True)


def _compute_risk_stats(
    sdf: pd.DataFrame,
    bdf: pd.DataFrame,
    risk_free_rate: float,
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
    var_s = float(rs_a.var())
    var_b = float(rb_a.var())
    # A flat return series (zero variance) makes every relative metric degenerate:
    # correlation collapses to 0/0 -> NaN, beta reads a spurious 0, sharpe divides by
    # a zero vol. It happens when a stock is locked limit-up/down across the whole
    # window (daily return is constant) or the benchmark is degenerate. Emitting
    # numbers here misdescribes the security — and the NaN would serialize to invalid
    # JSON — so refuse loudly, the same fail-loud policy as the _MIN_TRADING_DAYS floor.
    if var_s <= ZERO_THRESHOLD or var_b <= ZERO_THRESHOLD:
        msg = (
            "zero-variance return series: stock or benchmark is flat across the window "
            f"(suspected consecutive limit-up/down or a halt; var_stock={var_s:.3g}, "
            f"var_bench={var_b:.3g}); relative risk metrics are undefined, try another window"
        )
        raise ValueError(msg)
    beta = float(rs_a.cov(rb_a)) / var_b
    correlation = float(rs_a.corr(rb_a))

    excess = rs_a - rb_a
    tracking_error = float(excess.std()) * sqrt_252
    # None, not 0.0: tracking_error == 0 means the stock IS the benchmark
    # (rs_a == rb_a), so excess return / tracking error is 0/0 — undefined,
    # not "tracks benchmark perfectly with zero alpha". 0.0 reads as a real
    # active-management verdict; None says the metric isn't applicable.
    info_ratio = (ann_r_stock - ann_r_bench) / tracking_error if tracking_error > ZERO_THRESHOLD else None

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

        Returns one row per trading day in [start_date, end_date]. Warmup is
        auto-prefetched, so values inside the range are not null from warmup.

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
        elif not indicators:
            # Same trap as get_moving_averages: [] slips past the None check and the
            # unknown-name check below, yielding a silent date/close-only frame. Refuse it.
            msg = "indicators must not be empty; omit it for the default set or pass explicit names"
            raise ValueError(msg)
        requested = {i.upper() for i in indicators}
        # Fail loud on unknown names instead of silently dropping them.
        unknown = requested - set(_INDICATORS)
        if unknown:
            msg = f"unknown indicators {sorted(unknown)}; valid: {sorted(_INDICATORS)}"
            raise ValueError(msg)
        return _finalize(_apply_indicators(bs, code, start_date, end_date, requested))

    app.tool()(get_technical_indicators)


def _register_moving_averages(app: FastMCP, bs: Baostock) -> None:
    def get_moving_averages(
        code: str,
        start_date: str,
        end_date: str,
        periods: list[int] | None = None,
    ) -> list[dict[str, object]]:
        """Calculate SMA and EMA for multiple periods over the requested date range.

        Returns one row per trading day in [start_date, end_date]. Warmup is
        auto-prefetched, so values inside the range are not null from warmup.
        A period exceeding the stock's available history yields
        present-but-all-null SMA_<p>/EMA_<p> columns, so a requested period
        never silently vanishes.

        Args:
            code: Stock code.
            start_date: 'YYYY-MM-DD'.
            end_date: 'YYYY-MM-DD'.
            periods: Period list (each must be >= 1), e.g. [5,10,20,50,120,250]. Defaults to common set.

        """
        if periods is None:
            periods = list(_DEFAULT_MA_PERIODS)
        elif not periods:
            # [] is not None, so it skips the default above; left alone it would also
            # clear the `< 1` check (nothing to scan) and the for-loop (nothing to
            # iterate), silently returning a date/close-only frame with zero MA columns.
            # Refuse it like every other bad parameter — omit for default, or be explicit.
            msg = "periods must not be empty; omit it for the default set or pass explicit periods"
            raise ValueError(msg)
        invalid = [p for p in periods if p < 1]
        if invalid:
            msg = f"periods must be >= 1; got {invalid}"
            raise ValueError(msg)

        # Fetch enough warmup before start_date that every requested SMA(p)/EMA(p)
        # has reached its non-null phase before the user's window. 2x the max
        # period parallels the indicator side: SMA(p) is exact at p bars and EMA(p)
        # is still converging through its IIR recursion, so double for headroom.
        fetch_start = _expand_start(start_date, 2 * max(periods))
        df = _fetch_ohlcv(bs, code, fetch_start, end_date)
        result = df[["date", "close"]].copy()
        n = len(df)
        for p in periods:
            if n >= p:
                result[f"SMA_{p}"] = df["close"].rolling(window=p).mean()
                # adjust=False uses the recursive EMA formula
                # (EMA_t = alpha*x_t + (1-alpha)*EMA_{t-1}), matching ta's
                # internal MACD implementation and TradingView / talib convention.
                # min_periods=p forces the first p-1 un-warmed-up rows to NaN so EMA
                # shares SMA's warmup boundary and honors this tool's null-warmup
                # contract; without it ewm() emits a (misleading) value from row 0.
                result[f"EMA_{p}"] = df["close"].ewm(span=p, adjust=False, min_periods=p).mean()
            else:
                # Fewer bars than period p across the entire fetched history
                # (warmup + user window combined) — typical case is a 250-day
                # MA on a stock that IPO'd < 250 trading days ago. Emit explicit
                # all-null columns instead of dropping them, so a requested
                # period never just vanishes.
                result[f"SMA_{p}"] = np.nan
                result[f"EMA_{p}"] = np.nan

        user_start = pd.to_datetime(start_date)
        result = result[result["date"] >= user_start].reset_index(drop=True)
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
                Must yield >= 30 trading-day bars (~45+ calendar days, more if the
                span crosses a long holiday) or the call is rejected.
                ~245 calendar days ≈ 1 trading year (CN A-share).
            risk_free_rate: Annualized risk-free rate for Sharpe ratio.
                Default ~3% (approximate CN 10Y bond yield); override for non-CN markets.

        """
        start_date, end_date = lookback_range(lookback_days)

        sdf = _fetch_ohlcv(bs, code, start_date, end_date)
        bdf = _fetch_ohlcv(bs, benchmark_code, start_date, end_date)

        # Floor on real trading-day bars, not calendar days: a calendar window
        # can't guarantee a bar count (a span over Spring Festival yields far
        # fewer). Both legs must clear 30 — the CLT minimum for variance / cov.
        bars = min(len(sdf), len(bdf))
        if bars < _MIN_TRADING_DAYS:
            msg = (
                f"need >= {_MIN_TRADING_DAYS} trading days for risk metrics, got {bars} "
                f"(stock={len(sdf)}, benchmark={len(bdf)}); increase lookback_days"
            )
            raise ValueError(msg)

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
