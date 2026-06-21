"""Valuation analysis tools: PE/PB/PS, PEG, DDM, DCF, industry comparison, snapshot."""

from __future__ import annotations

import datetime as dt
from itertools import pairwise
from typing import TYPE_CHECKING

import numpy as np

from ashare_mcp.derivations import derive_ocf, get_latest_close
from ashare_mcp.errors import AkshareError, BaostockError, NoDataFoundError
from ashare_mcp.utils import MARKET_TZ, ZERO_THRESHOLD, Record, as_float, lookback_range, safe_float

if TYPE_CHECKING:
    import pandas as pd
    from mcp.server.fastmcp import FastMCP

    from ashare_mcp.akshare_source import AkshareSource
    from ashare_mcp.baostock_client import Baostock, BsQueryFn

VALUATION_FIELDS = "date,code,close,peTTM,pbMRQ,psTTM,pcfNcfTTM"

_MIN_DIVIDEND_YEARS = 2
_MIN_OCF_YEARS = 2
_DCF_HISTORY_YEARS = 5
_MAX_FORECAST_YEARS = 20
_MAX_FORECAST_GROWTH = 0.20
_MAX_DCF_GROWTH = 0.15
_MIN_DCF_GROWTH = -0.05
_COMPARISON_LOOKBACK_DAYS = 15
_SNAPSHOT_LOOKBACK_DAYS = 15
_PEG_LOOKBACK_DAYS = 30
# baostock query_profit_data is cumulative-from-year-start: Q=1 is 3-month,
# Q=2 is 6-month (H1), Q=3 is 9-month, Q=4 is 12-month (FY). Snapshot scans
# (year, quarter) in reverse, pruned by today's date so quarters whose
# disclosure window has not opened are skipped (see _profit_candidates).
_PROFIT_FALLBACK_YEARS = 3
_PERIOD_TYPE_BY_QUARTER: dict[int, str] = {1: "Q1", 2: "H1", 3: "9M", 4: "FY"}
_MONTHS_PER_QUARTER = 3


def _profit_candidates(
    today_year: int,
    today_month: int,
    fallback_years: int,
) -> list[tuple[int, int]]:
    """Reverse-chronological (year, quarter) candidates for the latest report.

    Prunes quarters whose CN A-share disclosure window has not opened:
      - Q1 disclosure opens Apr 1, H1 opens Jul 1, Q3 opens Oct 1
      - The annual report (Q4) is ALWAYS published in year Y+1, so the
        current year's Q4 is never a candidate.

    Total coverage spans `fallback_years` years (current year + the prior
    fallback_years - 1). Returns at most fallback_years * 4 - 1 pairs
    (worst case: month >= 10, the current year contributes Q3/Q2/Q1).

    Pure function: takes (year, month) ints so it's deterministic and
    trivially testable.
    """
    # (month - 1) // 3 maps 1-3 -> 0, 4-6 -> 1, 7-9 -> 2, 10-12 -> 3
    # Upper bound 3 is the structural proof that current-year Q4 never appears.
    max_q_this_year = (today_month - 1) // 3
    pairs = [(today_year, q) for q in range(max_q_this_year, 0, -1)]
    pairs.extend((today_year - off, q) for off in range(1, fallback_years) for q in (4, 3, 2, 1))
    return pairs


def _latest_profit(bs: Baostock, code: str) -> tuple[Record, int] | None:
    """Most recently published profit report as (record, quarter), or None.

    Scans (year, quarter) candidates newest-first; the first published report
    wins, so callers never depend on a specific (year, quarter) being available.
    Returns None when no candidate exists (suspended / fresh IPO / data delay).
    """
    now = dt.datetime.now(tz=MARKET_TZ)
    for year, quarter in _profit_candidates(now.year, now.month, _PROFIT_FALLBACK_YEARS):
        try:
            return bs.query_one("query_profit_data", code=code, year=str(year), quarter=quarter), quarter
        except NoDataFoundError:
            continue
    return None


def _latest_growth(bs: Baostock, code: str) -> tuple[Record, int] | None:
    """Most recently published growth report as (record, quarter), or None.

    Same disclosure-window scan as _latest_profit (growth_data shares the report
    calendar), so PEG pairs the current PE with the latest published growth
    rather than an arbitrary caller-supplied historical quarter.
    """
    now = dt.datetime.now(tz=MARKET_TZ)
    for year, quarter in _profit_candidates(now.year, now.month, _PROFIT_FALLBACK_YEARS):
        try:
            return bs.query_one("query_growth_data", code=code, year=str(year), quarter=quarter), quarter
        except NoDataFoundError:
            continue
    return None


# Fraction (0..1) -> percent-basis number. Used both for percentile rank and for
# converting baostock YOY* ratios (e.g. 0.18) to the percent number (18) that
# the PEG convention expects.
_PERCENT_SCALE = 100


def _clamp(value: float, lo: float, hi: float) -> tuple[float, str | None]:
    """Clamp value to [lo, hi]. Returns (clamped, reason_if_clamped)."""
    if value > hi:
        return hi, "exceeded_max"
    if value < lo:
        return lo, "below_min"
    return value, None


def _validate_forecast_years(forecast_years: int) -> None:
    """Reject forecast_years outside [1, _MAX_FORECAST_YEARS].

    Industry practice for the DCF/DDM explicit forecast period:
      - mature/stable companies: 5 years
      - moderate growth: 5-10 years
      - high growth: 7-10 years
      - hyper-growth (e.g. Uber pre-IPO): 10-15 years, up to 20 in extreme cases
    Beyond ~15-20 years the meaningfulness of individual-year cash-flow forecasts
    collapses (competition, technology, macro shifts dominate), and the standard
    move is to extend the convergence-to-terminal phase, not the explicit period.
    Capping at 20 years lets a caller cover the Uber-style outlier while rejecting
    obviously absurd horizons (e.g. 100 years would yield intrinsic values in the
    millions for a 20%-growth-clamped company).
    """
    if not 1 <= forecast_years <= _MAX_FORECAST_YEARS:
        msg = f"forecast_years ({forecast_years}) must be in [1, {_MAX_FORECAST_YEARS}]"
        raise ValueError(msg)


def _positive_stats(vals: pd.Series[float], current: float) -> dict[str, float | int | None]:
    """positive_{mean,median,min,max}/percentile over POSITIVE values only.

    A negative valuation multiple has no meaning -- a negative PE means the
    company is loss-making (not cheap), a negative P/CF means it burns cash --
    so a single negative observation poisons the mean and skews the percentile
    rank. We restrict the stats to the positive subset, the same positives-only
    contract compare_industry_valuation uses, and report `sample_size` so the
    caller sees how many observations fed them. `percentile_pct` is defined only
    when `current` itself is positive (a non-positive current can't be ranked
    within a positive distribution); `current`/`as_of` upstream still report the
    true latest value, negative or not.

    The stat keys carry a `positive_` prefix so the schema itself flags that
    these are NOT directly comparable to `current` when `current` is negative
    (a loss-making PE, a cash-burning P/CF). Without that prefix a caller
    would read `current=-467, positive_min=517` as "below historical floor" --
    the opposite of the truth -- because the field names hid that the stats
    were taken from a different sign-restricted distribution.
    """
    pos = vals[vals > 0]
    if pos.empty:
        return {
            "positive_mean": None,
            "positive_median": None,
            "positive_min": None,
            "positive_max": None,
            "percentile_pct": None,
            "sample_size": 0,
        }
    pct = float(np.mean(pos <= current)) * _PERCENT_SCALE if current > 0 else None
    return {
        "positive_mean": float(pos.mean()),
        "positive_median": float(pos.median()),
        "positive_min": float(pos.min()),
        "positive_max": float(pos.max()),
        "percentile_pct": pct,
        "sample_size": int(pos.size),
    }


def register(app: FastMCP, bs: Baostock, src: AkshareSource | None = None) -> None:
    """Register valuation tools with the MCP app."""
    _register_valuation_metrics(app, bs)
    _register_peg(app, bs)
    _register_ddm(app, bs)
    _register_dcf(app, bs, src)
    _register_industry_comparison(app, bs)
    _register_snapshot(app, bs)


def _register_valuation_metrics(app: FastMCP, bs: Baostock) -> None:
    def get_valuation_metrics(
        code: str,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> dict[str, object]:
        """Fetch valuation metrics (PE/PB/PS/PCF) history and current snapshot.

        Each metric in `metrics` independently reports `current` and `as_of`
        (the date of its last valid observation), since close and PE/PB/PS can be
        missing on different days. `period.last_trading_date` reports the last
        K-line bar in the window, separate from any metric's as_of.

        Each non-close metric also reports positive_mean/positive_median/
        positive_min/positive_max plus `percentile_pct` (0-100 rank of `current`:
        a fresh low ~0.4, an all-time high ~100), computed over POSITIVE values
        only. `sample_size` is the positive-observation count; `percentile_pct`
        is null when `current` is non-positive.

        Args:
            code: Stock code.
            start_date: Optional, defaults to 1 year ago.
            end_date: Optional, defaults to today.

        """
        now = dt.datetime.now(tz=MARKET_TZ)
        if end_date is None:
            end_date = now.strftime("%Y-%m-%d")
        if start_date is None:
            start_date = (now - dt.timedelta(days=365)).strftime("%Y-%m-%d")

        df = bs.query(
            "query_history_k_data_plus",
            code=code,
            fields=VALUATION_FIELDS,
            start_date=start_date,
            end_date=end_date,
            frequency="d",
            adjustflag="3",
        )

        metrics: dict[str, dict[str, float | int | str | None]] = {}
        for col in ["close", "peTTM", "pbMRQ", "psTTM", "pcfNcfTTM"]:
            vals = df[col].dropna()
            # baostock 0-fills a multiple the instrument lacks (an index has no
            # PE/PB/PS); treat that sentinel as no observation, like NaN, so an
            # index reports current/as_of None rather than a misleading 0 sitting
            # next to sample_size 0. close is a price, never a multiple — kept.
            if col != "close":
                vals = vals[vals.abs() >= ZERO_THRESHOLD]
            if vals.empty:
                metrics[col] = {"current": None, "as_of": None}
                continue
            current = float(vals.iloc[-1])
            entry: dict[str, float | int | str | None] = {
                "current": current,
                "as_of": str(df.loc[vals.index[-1], "date"]),
            }
            # close is a price, not a valuation multiple, so only current/as_of.
            # _positive_stats restricts positive_{mean,median,min,max}/percentile to
            # positive values; uses <= so a fresh low ranks at 1/n*100 (~0.4), not 0.
            if col != "close":
                entry.update(_positive_stats(vals, current))
            metrics[col] = entry

        return {
            "code": code,
            "period": {
                "start": start_date,
                "end": end_date,
                "trading_days": len(df),
                "last_trading_date": str(df.iloc[-1]["date"]),
            },
            "metrics": metrics,
        }

    app.tool()(get_valuation_metrics)


def _register_peg(app: FastMCP, bs: Baostock) -> None:
    def calculate_peg_ratio(code: str) -> dict[str, object]:
        """Calculate PEG = current PE_TTM / G, with G the latest published YoY net profit growth.

        PE is the latest available quote; G is the most recently disclosed YoY
        growth, returned as `growth_period` (statDate) and `growth_period_type`
        (Q1/H1/9M/FY). baostock's YOYNI is a ratio (0.1858 = +18.58% YoY); PEG
        uses the percent number (18.58).

        PEG is undefined and returns peg=None when:
          - PE == 0 (no TTM earnings data: index / non-equity, not loss-making)
          - PE < 0 (loss-making company; PE has no valuation meaning)
          - no growth report has been published yet (fresh IPO / data delay)
          - YoY growth <= 0 (PEG is a growth-stock metric; declining earnings
            yield a negative or nonsensical value that LLMs would misread as cheap)

        Args:
            code: Stock code.

        """
        start, end = lookback_range(_PEG_LOOKBACK_DAYS)
        price_data = bs.query(
            "query_history_k_data_plus",
            code=code,
            fields="date,close,peTTM",
            start_date=start,
            end_date=end,
            frequency="d",
            adjustflag="3",
        )
        pe_series = price_data["peTTM"].dropna()
        if pe_series.empty:
            return {
                "code": code,
                "pe_ttm": None,
                "yoyni": None,
                "peg": None,
                "reason": "no PE available (likely loss-making or recently suspended)",
            }
        pe = float(pe_series.iloc[-1])
        if pe == 0:
            # Not a real PE (would need infinite earnings); baostock fills 0 for
            # indices / instruments without TTM earnings data, not loss-making.
            return {
                "code": code,
                "pe_ttm": pe,
                "yoyni": None,
                "peg": None,
                "reason": "PE is 0: no TTM earnings data (index / non-equity instrument)",
            }
        if pe < 0:
            return {
                "code": code,
                "pe_ttm": pe,
                "yoyni": None,
                "peg": None,
                "reason": "PEG undefined for negative PE (loss-making company)",
            }

        latest = _latest_growth(bs, code)
        if latest is None:
            return {
                "code": code,
                "pe_ttm": pe,
                "yoyni": None,
                "peg": None,
                "reason": "no growth report published yet (fresh IPO or data delay)",
            }
        growth, g_quarter = latest
        growth_period = growth.get("statDate")
        growth_period_type = _PERIOD_TYPE_BY_QUARTER[g_quarter]
        yoyni_raw = safe_float(growth.get("YOYNI"))
        if yoyni_raw is None or yoyni_raw <= 0:
            return {
                "code": code,
                "pe_ttm": pe,
                "yoyni": yoyni_raw,
                "peg": None,
                "growth_period": growth_period,
                "growth_period_type": growth_period_type,
                "reason": "PEG undefined for non-positive growth",
            }

        # yoyni_raw > 0 here, so growth_pct > 0 — division is safe without ZERO_THRESHOLD guard.
        growth_pct = yoyni_raw * _PERCENT_SCALE
        return {
            "code": code,
            "pe_ttm": pe,
            "yoyni": yoyni_raw,
            "yoyni_pct": growth_pct,
            "growth_period": growth_period,
            "growth_period_type": growth_period_type,
            "peg": pe / growth_pct,
        }

    app.tool()(calculate_peg_ratio)


def _collect_annual_dividends(bs: Baostock, code: str, current_year: int, years_back: int) -> dict[int, float]:
    """Sum announced cash dividends per year over the lookback window.

    Stops BEFORE current_year on purpose: baostock buckets dividends by announcement
    year, so mid-year the current bucket is usually partial (e.g. a bank's interim
    payout not yet declared); using it as divs[-1] would understate the DDM base.
    The same bucketing can also split one fiscal year across adjacent buckets --
    a baostock granularity limit, not corrected here.
    """
    annual_divs: dict[int, float] = {}
    for y in range(current_year - years_back, current_year):
        try:
            df = bs.query("query_dividend_data", code=code, year=str(y), yearType="report")
        except NoDataFoundError:
            continue
        vals = df["dividCashPsBeforeTax"].dropna()
        if not vals.empty:
            # round at the sum site: pandas sum() over already-rounded payout
            # floats reintroduces IEEE 754 noise (27.673 + 23.957 =
            # 51.629999999999995), and this dict feeds DDM dividend_history
            # directly without going through the df_to_records / scalar() pipeline.
            annual_divs[y] = round(float(vals.sum()), 10)
    return annual_divs


def _register_ddm(app: FastMCP, bs: Baostock) -> None:
    def calculate_ddm_valuation(
        code: str,
        discount_rate: float,
        terminal_growth_rate: float,
        years_back: int = 5,
        forecast_years: int = 5,
    ) -> dict[str, object]:
        """DDM (Dividend Discount Model) valuation.

        Uses dividCashPsBeforeTax from baostock. Auto-sums semicolon-separated multi-payouts.
        Current price from latest K-line close. Excludes the current calendar year,
        whose dividend bucket is usually incomplete mid-year. Buckets are by
        announcement year, so a prior fiscal year's final payout and the next year's
        interim can land in the same bucket, distorting dividend_cagr.

        `forecast_growth` caps `dividend_cagr` at `growth_clamp_bounds.max` (20%);
        negative CAGR passes through unchanged. `growth_clamped` is "exceeded_max"
        if the cap fired, else null.

        Args:
            code: Stock code.
            discount_rate: Required rate of return, e.g. 0.10.
            terminal_growth_rate: Perpetual growth rate, e.g. 0.025.
            years_back: Years of dividend history to use.
            forecast_years: Projection period (must be in [1, 20]).

        """
        if discount_rate <= terminal_growth_rate:
            msg = (
                f"discount_rate ({discount_rate}) must exceed terminal_growth_rate "
                f"({terminal_growth_rate}); Gordon model requires r > g"
            )
            raise ValueError(msg)
        _validate_forecast_years(forecast_years)
        # years_back <= 0 makes range(...) empty and the loop silently yields
        # "insufficient dividend history" — the same fail-loud policy as every
        # other parameter (forecast_years, periods, indicators) requires raising.
        if years_back < _MIN_DIVIDEND_YEARS:
            msg = (
                f"years_back ({years_back}) must be >= {_MIN_DIVIDEND_YEARS}; "
                "DDM CAGR needs at least two annual buckets"
            )
            raise ValueError(msg)

        current_year = dt.datetime.now(tz=MARKET_TZ).year
        annual_divs = _collect_annual_dividends(bs, code, current_year, years_back)

        if len(annual_divs) < _MIN_DIVIDEND_YEARS:
            return {"code": code, "error": "insufficient dividend history", "years_found": len(annual_divs)}

        years = sorted(annual_divs.keys())
        divs = [annual_divs[y] for y in years]

        # CAGR (geometric mean): consistent with DCF and correct for compound
        # growth. Arithmetic mean would massively overstate growth on a volatile
        # dividend series (e.g. [1,2,1,2,1] arithmetic = +25% but true CAGR = 0%).
        # divs[0] <= ZERO_THRESHOLD means the series starts at zero — CAGR is undefined; we
        # fall back to 0 growth, which is the conservative assumption for DDM.
        # Calendar span, not len(divs)-1: gap years are absent from annual_divs and would inflate CAGR.
        n_periods = years[-1] - years[0]
        cagr = (divs[-1] / divs[0]) ** (1 / n_periods) - 1 if divs[0] > ZERO_THRESHOLD and n_periods > 0 else 0.0
        # Cap only the upside: an unsustainably high historical CAGR projected forever
        # blows up the terminal value, so trim it. Negative CAGR is left alone — GGM
        # is well-defined for g < 0 and yields the correctly lower valuation for a
        # company with shrinking dividends; flooring it would inflate dying stocks.
        if cagr > _MAX_FORECAST_GROWTH:
            forecast_growth, growth_clamped = _MAX_FORECAST_GROWTH, "exceeded_max"
        else:
            forecast_growth, growth_clamped = cagr, None

        latest_div = divs[-1]
        pv_divs: list[dict[str, float]] = []
        for i in range(1, forecast_years + 1):
            # round at the arithmetic site: power+multiply chains accumulate IEEE 754
            # noise that this dict ships verbatim (no df_to_records pipeline).
            future_div = round(latest_div * (1 + forecast_growth) ** i, 10)
            pv = round(future_div / (1 + discount_rate) ** i, 10)
            pv_divs.append({"year": float(years[-1] + i), "dividend": future_div, "pv": pv})

        terminal_div = latest_div * (1 + forecast_growth) ** forecast_years * (1 + terminal_growth_rate)
        terminal_value = terminal_div / (discount_rate - terminal_growth_rate)
        pv_terminal = round(terminal_value / (1 + discount_rate) ** forecast_years, 10)

        result: dict[str, object] = {
            "code": code,
            "intrinsic_value_per_share": round(sum(d["pv"] for d in pv_divs) + pv_terminal, 10),
            "current_price": get_latest_close(bs, code),
            "dividend_history": {str(y): d for y, d in zip(years, divs, strict=True)},
            "dividend_cagr": round(cagr, 10),
            "forecast_growth": round(forecast_growth, 10),
            "growth_clamped": growth_clamped,
            "growth_clamp_bounds": {"max": _MAX_FORECAST_GROWTH},
            "projected_dividends": pv_divs,
            "pv_terminal": pv_terminal,
            "assumptions": {
                "discount_rate": discount_rate,
                "terminal_growth_rate": terminal_growth_rate,
                "forecast_years": forecast_years,
            },
        }
        # Cost of equity = risk-free + ERP; for CN A-shares both are positive, so a
        # non-positive discount_rate is almost always a user input error rather than
        # an intentional NIRP/social-discount setup. Don't reject (those use cases
        # are mathematically valid) — surface a warning and let the caller decide.
        if discount_rate <= 0:
            result["discount_rate_warning"] = (
                f"discount_rate={discount_rate} is non-positive; unusual for equity "
                "valuation (cost of equity = risk-free + ERP is normally positive), "
                "check input"
            )
        return result

    app.tool()(calculate_ddm_valuation)


def _check_fcf_endpoints(
    code: str,
    years: list[int],
    fcfs: list[float],
) -> dict[str, object] | None:
    """Return an early-exit error dict if FCF endpoints invalidate DCF, else None.

    Both endpoints must be positive: fcfs[0] anchors CAGR's denominator, fcfs[-1]
    anchors the perpetuity projection. A non-positive last-year FCF would project
    as a negative perpetual cash flow — mathematically yields an EV but has no
    economic meaning ("company burns cash forever").
    """
    non_positive: list[str] = []
    if fcfs[0] <= 0:
        non_positive.append(f"first-year ({fcfs[0]:.3g})")
    if fcfs[-1] <= 0:
        non_positive.append(f"last-year ({fcfs[-1]:.3g})")
    if not non_positive:
        return None
    return {
        "code": code,
        "error": f"FCF non-positive at {' and '.join(non_positive)}; DCF undefined",
        "fcf_history": {str(y): v for y, v in zip(years, fcfs, strict=True)},
    }


def _gather_fcf_baostock(
    bs: Baostock,
    code: str,
    capex_to_ocf_ratio: float,
) -> tuple[list[int], list[float], list[float], float | None, dict[str, object]]:
    """Gather FCF history via baostock estimation (MBRevenue * CFOToOR).

    Resolves total_shares from the last year's profit record — already fetched
    here, so no extra query — and returns it directly. The raw profit records
    stay internal, so callers can't mistake their presence for a path flag.
    """
    current_year = dt.datetime.now(tz=MARKET_TZ).year
    profit_history: dict[int, Record] = {}
    ocf_history: dict[int, float] = {}
    for y in range(current_year - _DCF_HISTORY_YEARS, current_year + 1):
        try:
            profit = bs.query_one("query_profit_data", code=code, year=str(y), quarter=4)
            cash_flow = bs.query_one("query_cash_flow_data", code=code, year=str(y), quarter=4)
        except NoDataFoundError:
            continue
        try:
            ocf, _ = derive_ocf(profit, cash_flow)
        except (TypeError, ValueError):
            continue
        profit_history[y] = profit
        ocf_history[y] = ocf
    years = sorted(ocf_history.keys())
    # round at the source: derive_ocf is MBRevenue * CFOToOR (multiplication noise),
    # and fcf is another multiplication on top. Downstream dicts (fcf_history,
    # ocf_history, projected) ship these verbatim outside the scalar() pipeline.
    ocfs = [round(ocf_history[y], 10) for y in years]
    fcfs = [round(o * (1 - capex_to_ocf_ratio), 10) for o in ocfs]
    # years[-1] is in profit_history (co-indexed: a year enters only if both queries succeeded).
    total_shares = safe_float(profit_history[years[-1]]["totalShare"]) if years else None
    provenance: dict[str, object] = {
        "ocf": {"derived": True, "formula": "MBRevenue * CFOToOR", "approx_precision_pct": 2},
        "capex": {"user_provided": True, "capex_to_ocf_ratio": capex_to_ocf_ratio},
    }
    return years, ocfs, fcfs, total_shares, provenance


def _gather_fcf_akshare(
    src: AkshareSource,
    code: str,
) -> tuple[list[int], list[float], list[float], dict[str, object]]:
    """Gather FCF history via akshare real data (NETCASH_OPERATE - CONSTRUCT_LONG_ASSET)."""
    history = src.ocf_and_capex_history(code, _DCF_HISTORY_YEARS)
    years = sorted(history.keys())
    # akshare values come in already rounded, but the subtraction can still
    # introduce noise (1.5e10 - 3.1e9 hits FPU precision limits); round
    # to match the baostock path's contract.
    ocfs = [round(history[y][0], 10) for y in years]
    fcfs = [round(history[y][0] - history[y][1], 10) for y in years]
    provenance: dict[str, object] = {
        "ocf": {"derived": False, "source": "akshare", "field": "NETCASH_OPERATE"},
        "capex": {"derived": False, "source": "akshare", "field": "CONSTRUCT_LONG_ASSET"},
    }
    return years, ocfs, fcfs, provenance


def _gather_fcf(
    bs: Baostock,
    code: str,
    src: AkshareSource | None,
    capex_to_ocf_ratio: float,
) -> tuple[list[int], list[float], list[float], float | None, dict[str, object], bool]:
    """Gather FCF history, preferring akshare real data over baostock estimation.

    Falls back to baostock when akshare is absent, errors, or returns fewer than
    _MIN_OCF_YEARS years — a partial akshare result must not block a baostock path
    that may carry the full history. capex_to_ocf_ratio is always required; on the
    akshare-success path it is simply unused (real capex wins), so the baostock
    fallback can never be stranded without a ratio at runtime.

    Returns (years, ocfs, fcfs, total_shares, provenance, used_akshare); total_shares
    is non-None only on the baostock path (free from its profit fetch).
    """
    years: list[int] = []
    ocfs: list[float] = []
    fcfs: list[float] = []
    provenance: dict[str, object] = {}

    akshare_skip: str | None
    if src is None:
        akshare_skip = "akshare not installed"
    else:
        try:
            years, ocfs, fcfs, provenance = _gather_fcf_akshare(src, code)
            akshare_skip = None
        except AkshareError as e:
            if e.no_data:
                # akshare serves other firms but returned nothing for this one; the
                # MBRevenue * CFOToOR estimate is no more valid here, so don't fall
                # back to it. Propagate; calculate_dcf_valuation returns a clean error.
                raise
            akshare_skip = f"akshare failed: {e}"
        if akshare_skip is None and len(years) < _MIN_OCF_YEARS:
            akshare_skip = f"akshare returned {len(years)} year(s), need >= {_MIN_OCF_YEARS}"
    if akshare_skip is None:
        return years, ocfs, fcfs, None, provenance, True

    years, ocfs, fcfs, total_shares, provenance = _gather_fcf_baostock(bs, code, capex_to_ocf_ratio)
    if src is not None:
        provenance["akshare_fallback"] = akshare_skip
    return years, ocfs, fcfs, total_shares, provenance, False


def _resolve_net_debt(
    net_debt: float | None,
    src: AkshareSource | None,
    code: str,
) -> tuple[float | None, dict[str, object]]:
    """Resolve net_debt value and its provenance."""
    if net_debt is not None:
        return net_debt, {"user_provided": True}
    if src is not None:
        try:
            nd_result = src.net_debt(code)
            return as_float(nd_result["net_debt"]), {"auto_computed": True, "source": "akshare", "details": nd_result}
        except AkshareError:
            return None, {"user_provided": False, "note": "akshare balance sheet unavailable; pass net_debt manually"}
    return None, {"user_provided": False, "note": "baostock provides no interest-bearing debt or cash breakdown"}


def _resolve_equity(ev: float, net_debt: float | None, total_shares: float | None) -> dict[str, object]:
    """Compute equity fields or explain why they're missing.

    Pure function: total_shares is resolved by the caller (baostock path reuses
    its already-fetched profit record; akshare path calls _latest_profit), so
    this never needs to know which data source was used.
    """
    if net_debt is None:
        return {
            "equity_note": (
                "per_share_intrinsic_value omitted: net_debt not provided. "
                "Pass net_debt (interest-bearing debt minus cash) to derive equity value."
            ),
        }
    # round subtraction / division at the site: ev is already rounded upstream,
    # but ev - net_debt and the per-share division can both reintroduce noise
    # (subtraction of close magnitudes, division at FPU precision limits).
    equity_value = round(ev - net_debt, 10)
    if total_shares is None or total_shares <= ZERO_THRESHOLD:
        return {
            "net_debt": net_debt,
            "equity_value": equity_value,
            "total_shares": total_shares,
            "equity_note": "per_share_intrinsic_value omitted: total shares unavailable",
        }
    return {
        "net_debt": net_debt,
        "equity_value": equity_value,
        "total_shares": total_shares,
        "per_share_intrinsic_value": round(equity_value / total_shares, 10),
    }


def _resolve_total_shares(
    bs: Baostock,
    code: str,
    baostock_shares: float | None,
    *,
    used_akshare: bool,
) -> float | None:
    """Total shares outstanding for the equity step.

    The baostock path already pulled it from the gathered profit record (free);
    the akshare path never queried baostock, so fetch the latest report now.
    """
    if not used_akshare:
        return baostock_shares
    latest = _latest_profit(bs, code)
    return safe_float(latest[0]["totalShare"]) if latest is not None else None


def _register_dcf(app: FastMCP, bs: Baostock, src: AkshareSource | None = None) -> None:
    def calculate_dcf_valuation(
        code: str,
        discount_rate: float,
        terminal_growth_rate: float,
        capex_to_ocf_ratio: float,
        net_debt: float | None = None,
        forecast_years: int = 5,
    ) -> dict[str, object]:
        """Simplified DCF valuation. OCF derived as MBRevenue * CFOToOR (~2% precision).

        FCF = OCF * (1 - capex_to_ocf_ratio). Caller must supply capex_to_ocf_ratio
        because baostock provides no Capex data.

        Enterprise value (EV) is always returned. To derive equity value and
        per-share intrinsic value, caller must provide `net_debt`, which baostock
        does NOT publish — it must come from the balance sheet (interest-bearing
        debt minus cash & equivalents). Without it, only EV is returned and the
        caller can compute equity = ev - (their own net_debt).

        When akshare is installed and reachable, real operating cash flow
        (NETCASH_OPERATE) and real capex (CONSTRUCT_LONG_ASSET) are used
        automatically (capex_to_ocf_ratio is then ignored), and net_debt is computed
        from the balance sheet if not provided. If akshare is absent or fails at
        runtime (rate-limit, anti-scrape), the calculation falls back to baostock
        estimation using capex_to_ocf_ratio. data_provenance indicates which path.

        Uses a fixed 5-year FCF history, degrading gracefully to as few as 2
        years when reports are missing.

        `forecast_growth` clamps `fcf_cagr` to `growth_clamp_bounds` ([-5%, 15%]);
        `growth_clamped` reports which bound (if any) was hit. Read it before using
        `forecast_growth` -- a raw CAGR outside the band was silently overridden.

        `intermediates.fcf_quality` reports negative_years / sign_changes / min / max
        on the historical FCF series; fcf_cagr is endpoint-anchored and does not
        reflect intra-period volatility.

        Args:
            code: Stock code.
            discount_rate: WACC / discount rate, e.g. 0.10.
            terminal_growth_rate: Perpetual growth rate, e.g. 0.025.
            capex_to_ocf_ratio: Capex as a fraction of OCF. Always required, but
                used only on the baostock fallback path; ignored when akshare's
                real capex is available.
            net_debt: Interest-bearing debt minus cash & equivalents (CNY).
                Negative = net cash position. Required for per-share valuation.
            forecast_years: Projection period (must be in [1, 20]).

        """
        if discount_rate <= terminal_growth_rate:
            msg = (
                f"discount_rate ({discount_rate}) must exceed terminal_growth_rate "
                f"({terminal_growth_rate}); Gordon model requires r > g"
            )
            raise ValueError(msg)
        _validate_forecast_years(forecast_years)

        try:
            years, ocfs, fcfs, total_shares, provenance, used_akshare = _gather_fcf(
                bs,
                code,
                src,
                capex_to_ocf_ratio,
            )
        except AkshareError:
            return {"code": code, "error": "cash-flow data could not be retrieved"}

        if len(years) < _MIN_OCF_YEARS:
            return {"code": code, "error": "insufficient OCF history", "years_found": len(years)}

        endpoint_err = _check_fcf_endpoints(code, years, fcfs)
        if endpoint_err is not None:
            return endpoint_err

        ratio = fcfs[-1] / fcfs[0]
        # Calendar span, not len(fcfs)-1: a missing report year is absent from
        # years/fcfs (NoDataFoundError -> continue), so the array length
        # undercounts the elapsed years and would inflate CAGR. Mirrors the DDM fix.
        n = years[-1] - years[0]
        cagr = ratio ** (1 / n) - 1 if ratio > ZERO_THRESHOLD and n > 0 else 0.0
        forecast_growth, growth_clamped = _clamp(cagr, _MIN_DCF_GROWTH, _MAX_DCF_GROWTH)

        projected: list[dict[str, float]] = []
        last_fcf = fcfs[-1]
        for i in range(1, forecast_years + 1):
            # round at the arithmetic site, same reason as DDM: this dict ships
            # verbatim, bypassing the scalar() normalization pipeline.
            future_fcf = round(last_fcf * (1 + forecast_growth) ** i, 10)
            pv = round(future_fcf / (1 + discount_rate) ** i, 10)
            projected.append({"year": float(years[-1] + i), "fcf": future_fcf, "pv": pv})

        terminal_fcf = projected[-1]["fcf"] * (1 + terminal_growth_rate)
        terminal_value = terminal_fcf / (discount_rate - terminal_growth_rate)
        pv_terminal = round(terminal_value / (1 + discount_rate) ** forecast_years, 10)
        ev = round(sum(p["pv"] for p in projected) + pv_terminal, 10)

        net_debt, nd_provenance = _resolve_net_debt(net_debt, src, code)
        provenance["net_debt"] = nd_provenance

        result: dict[str, object] = {
            "code": code,
            "industry": str(bs.query_one("query_stock_industry", code=code)["industry"]),
            "current_price": get_latest_close(bs, code),
            "enterprise_value": ev,
            "intermediates": {
                "ocf_history": {str(y): v for y, v in zip(years, ocfs, strict=True)},
                "fcf_history": {str(y): v for y, v in zip(years, fcfs, strict=True)},
                "fcf_quality": {
                    "negative_years": [y for y, f in zip(years, fcfs, strict=True) if f <= 0],
                    "min": min(fcfs),
                    "max": max(fcfs),
                    "sign_changes": sum(1 for a, b in pairwise(fcfs) if (a > 0) != (b > 0)),
                },
                "fcf_cagr": round(cagr, 10),
                "forecast_growth": round(forecast_growth, 10),
                "growth_clamped": growth_clamped,
                "growth_clamp_bounds": {"min": _MIN_DCF_GROWTH, "max": _MAX_DCF_GROWTH},
            },
            "projected_fcf": projected,
            "pv_terminal": pv_terminal,
            "assumptions": {
                "discount_rate": discount_rate,
                "terminal_growth_rate": terminal_growth_rate,
                # akshare path: real capex won, the input ratio was overridden —
                # emit only `_ignored: true` so the caller sees their argument
                # was silently replaced by real data. baostock path: the ratio
                # IS the assumption, emit it as `capex_to_ocf_ratio: <value>`.
                # Exactly one key in each branch, no `null + ignored` pair.
                **(
                    {"capex_to_ocf_ratio_ignored": True} if used_akshare else {"capex_to_ocf_ratio": capex_to_ocf_ratio}
                ),
                "forecast_years": forecast_years,
            },
            "data_provenance": provenance,
        }

        total_shares = _resolve_total_shares(bs, code, total_shares, used_akshare=used_akshare)
        result.update(_resolve_equity(ev, net_debt, total_shares))
        return result

    app.tool()(calculate_dcf_valuation)


def _collect_peer_valuations(
    bs: Baostock,
    peers: pd.DataFrame,
    start_date: str,
    end_date: str,
    target_code: str,
) -> tuple[list[dict[str, object]], list[dict[str, str]]]:
    """Fetch each peer's latest valuation row; failures are bucketed into `skipped`.

    NoDataFoundError / BaostockError / empty frame each map to a distinct skip
    reason rather than aborting the whole comparison -- one dead peer must not
    sink the industry view.
    """
    valuations: list[dict[str, object]] = []
    skipped: list[dict[str, str]] = []
    for _, row in peers.iterrows():
        sc = str(row["code"])
        try:
            vdf = bs.query(
                "query_history_k_data_plus",
                code=sc,
                fields=VALUATION_FIELDS,
                start_date=start_date,
                end_date=end_date,
                frequency="d",
                adjustflag="3",
            )
        except NoDataFoundError:
            skipped.append({"code": sc, "reason": "no_data"})
            continue
        except BaostockError:
            skipped.append({"code": sc, "reason": "api_error"})
            continue
        if vdf.empty:
            skipped.append({"code": sc, "reason": "empty_result"})
            continue
        latest = vdf.iloc[-1]
        valuations.append(
            {
                "code": sc,
                "code_name": str(row.get("code_name", sc)),
                "close": safe_float(latest["close"]),
                "peTTM": safe_float(latest["peTTM"]),
                "pbMRQ": safe_float(latest["pbMRQ"]),
                "psTTM": safe_float(latest["psTTM"]),
                "is_target": sc == target_code,
            },
        )
    return valuations, skipped


_TRIM_PROPORTION = 0.10  # per-tail fraction dropped by the industry trimmed mean


def _trimmed_mean(vals: list[float], proportion: float = _TRIM_PROPORTION) -> float:
    """Symmetric trimmed mean: drop max(1, int(n*proportion)) values per tail.

    PE/PB/PS are right-skewed — a near-zero-earnings peer yields a PE in the
    thousands, dragging a plain mean far above the typical peer. Trimming both
    tails suppresses those extremes. At least one value per tail is always cut
    (industry peer sets run well above 5), so even a small peer set loses its
    single worst outlier; falls back to the full mean only when trimming would
    leave nothing (n <= 2).
    """
    arr = np.sort(np.array(vals))
    n = arr.size
    k = max(1, int(n * proportion))
    core = arr[k:-k]
    return float(core.mean()) if core.size else float(arr.mean())


def _industry_stats(valuations: list[dict[str, object]]) -> dict[str, dict[str, float]]:
    """trimmed_mean/median/min/max per multiple over POSITIVE peer values only.

    A negative PE means the peer is loss-making (no valuation meaning), not
    "cheap" -- so each metric filters to positives independently. The center stat
    is a 10%-per-tail trimmed mean (>=1 per tail), not the raw mean: PE/PB/PS are
    right-skewed and one near-zero-earnings peer (PE in the thousands) wrecks a
    plain average. `count` reports how many peers fed each metric.
    """
    stats: dict[str, dict[str, float]] = {}
    for metric in ["peTTM", "pbMRQ", "psTTM"]:
        vals_list = [f for v in valuations if (f := safe_float(v.get(metric))) is not None and f > 0]
        if vals_list:
            arr = np.array(vals_list)
            stats[metric] = {
                "count": len(vals_list),
                "trimmed_mean": _trimmed_mean(vals_list),
                "median": float(np.median(arr)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
            }
    return stats


def _register_industry_comparison(app: FastMCP, bs: Baostock) -> None:
    def compare_industry_valuation(code: str, date: str | None = None) -> dict[str, object]:
        """Compare a stock's valuation (PE/PB/PS) against its industry peers.

        Behavior contract:
          - Raises NoDataFoundError if `code` is absent from the industry table
            for `date`, has no industry classification (delisted / legacy stock),
            or has no K-line in the lookback window — it will not return a result
            without the target.
          - Peers whose K-line is missing are listed in `peer_coverage.skipped`
            with the reason. Industry statistics exclude the target itself and
            use POSITIVE values only; the center stat is `trimmed_mean` (10% per
            tail, >=1 each side), not a raw mean. `industry_stats[metric].count`
            reports peers per statistic.

        Args:
            code: Target stock code.
            date: Comparison date 'YYYY-MM-DD'. Defaults to latest.

        """
        # Single industry query: target's industry is filtered out of the full
        # snapshot, so target_industry and peers share the same date semantics
        # (previously they didn't — target was latest, peers respected `date`).
        industry_fn: BsQueryFn = "query_stock_industry"
        all_ind = bs.query(industry_fn, date=date)
        target_row = all_ind[all_ind["code"] == code]
        # Explicit object value type: NoDataFoundError takes dict[str, object]
        # and dict is invariant, so an inferred dict[str, str] won't unify.
        industry_ctx: dict[str, object] = {"code": code, "date": date or "latest"}
        if target_row.empty:
            # The query DID return rows (whole industry table) -- target is just
            # not one of them. The default `no rows returned from {fn}()` phrasing
            # would lie; pass `reason` to say what actually went wrong. `fn` stays
            # the real baostock call so the error layer doesn't impersonate one.
            raise NoDataFoundError(
                industry_fn,
                industry_ctx,
                reason=f"{code} not present in industry classification table for {date or 'latest'}",
            )
        target_industry = str(target_row.iloc[0]["industry"]).strip()
        if not target_industry:
            # Empty industry = delisted / legacy stock with no classification.
            # Matching on "" would lump in every other unclassified stock as a
            # bogus peer, so refuse — same fail-loud contract as target missing.
            raise NoDataFoundError(
                industry_fn,
                industry_ctx,
                reason=f"{code} has no industry classification (delisted or legacy stock)",
            )
        peers = all_ind[all_ind["industry"] == target_industry]

        start_date, end_date = lookback_range(_COMPARISON_LOOKBACK_DAYS, end=date)

        valuations, skipped = _collect_peer_valuations(bs, peers, start_date, end_date, code)

        # Target self-check: classification table said target exists, but its
        # K-line query may still have failed (suspension / IPO timing / etc).
        # Refuse to return a meaningful-looking result without the target.
        target_records = [v for v in valuations if v.get("is_target")]
        if not target_records:
            kline_fn: BsQueryFn = "query_history_k_data_plus"
            raise NoDataFoundError(
                kline_fn,
                {
                    "code": code,
                    "industry": target_industry,
                    "lookback_days": _COMPARISON_LOOKBACK_DAYS,
                    "end_date": end_date,
                    "peers_published": len(valuations),
                    "peers_skipped": len(skipped),
                },
                reason=(
                    f"{code} has no K-line in the {_COMPARISON_LOOKBACK_DAYS}-day "
                    f"lookback ending {end_date} (suspended / fresh IPO / data delay); "
                    f"industry {target_industry!r} has {len(valuations)} peers published"
                ),
            )

        stats = _industry_stats([v for v in valuations if not v.get("is_target")])

        return {
            "code": code,
            "industry": target_industry,
            "peer_coverage": {
                "total": len(peers),
                "published": len(valuations),
                "skipped": skipped,
            },
            "target": target_records[0],
            "industry_stats": stats,
            "peers": valuations,
        }

    app.tool()(compare_industry_valuation)


def _drop_zero_fill(v: object) -> float | None:
    """Map baostock's 0-fill for a PE/PB/PS multiple to None.

    baostock fills 0 for a valuation multiple an instrument cannot have (an index
    has no earnings/book/sales), which reads as a misleadingly cheap 0. Surface
    that sentinel as None; a genuine negative multiple (loss-making / negative
    equity) is real and kept.
    """
    f = safe_float(v)
    return None if f is not None and abs(f) < ZERO_THRESHOLD else f


def _optional_record(bs: Baostock, fn: BsQueryFn, code: str) -> Record:
    """query_one, but an empty dict instead of raising when there are no rows.

    Indices (sh.000300, etc.) carry K-line + valuation data but have no industry
    classification and no stock-basic row, so query_stock_industry / _basic
    return zero rows for them. The snapshot treats those as optional add-ons
    rather than aborting — price + valuation are its core, and an index
    legitimately lacks industry / IPO date / profit. Returning {} keeps the
    .get() chain at the call site uniform whether the row existed or not.
    """
    try:
        return bs.query_one(fn, code=code)
    except NoDataFoundError:
        return {}


def _register_snapshot(app: FastMCP, bs: Baostock) -> None:
    def get_stock_snapshot(code: str) -> dict[str, object]:
        """One-call snapshot: latest price, valuation, industry, total shares, net profit.

        Combines data from K-line, stock_industry, and profit_data.

        Designed for individual stocks. An index (sh.000300, etc.) has K-line +
        valuation but no industry / basic / profit rows; those fields come back
        null instead of aborting the call, so price + PE/PB/PS are still returned.

        Profit fields (`net_profit`, `revenue`) come from the most recently
        published report, so they may be Q1, H1, 9M, or FY depending on the
        calendar. Always inspect `profit_period_type` ('Q1'/'H1'/'9M'/'FY') and
        `profit_period_months` (3/6/9/12) before comparing across companies:
        baostock reports these cumulative-from-year-start, so Q1's net_profit is
        3 months, NOT comparable to FY's 12. To annualize, multiply Q1 by 4, H1 by 2.

        baostock fills MBRevenue only on H1 / FY reports, so `revenue` is null
        when `profit_period_type` is 'Q1' or '9M'.

        Args:
            code: Stock code.

        """
        # Single K-line query: VALUATION_FIELDS already contains `close`, so a
        # separate get_latest_close() call would just duplicate the network round-trip.
        start, end = lookback_range(_SNAPSHOT_LOOKBACK_DAYS)
        kdf = bs.query(
            "query_history_k_data_plus",
            code=code,
            fields=VALUATION_FIELDS,
            start_date=start,
            end_date=end,
            frequency="d",
            adjustflag="3",
        )
        latest_k = kdf.iloc[-1]
        price = safe_float(latest_k["close"])

        # Optional: an index has K-line + valuation but no industry / basic row,
        # so these return {} rather than aborting the whole snapshot.
        ind = _optional_record(bs, "query_stock_industry", code)
        basic = _optional_record(bs, "query_stock_basic", code)

        # Scan (year, quarter) in reverse chronological order, pruned to
        # quarters whose disclosure window has opened — the newest
        # published report wins regardless of which quarter it happens
        # to be. baostock returns cumulative YTD values; callers must
        # read profit_period_type before comparing across companies.
        latest = _latest_profit(bs, code)
        profit = latest[0] if latest is not None else None
        period_quarter = latest[1] if latest is not None else None

        result: dict[str, object] = {
            "code": code,
            "code_name": ind.get("code_name") or basic.get("code_name"),
            "industry": ind.get("industry"),
            "ipo_date": basic.get("ipoDate"),
            "close": price,
            "peTTM": _drop_zero_fill(latest_k["peTTM"]),
            "pbMRQ": _drop_zero_fill(latest_k["pbMRQ"]),
            "psTTM": _drop_zero_fill(latest_k["psTTM"]),
        }
        if profit and period_quarter is not None:
            result["total_shares"] = safe_float(profit["totalShare"])
            result["net_profit"] = safe_float(profit["netProfit"])
            result["revenue"] = safe_float(profit["MBRevenue"])
            result["profit_period"] = profit.get("statDate")
            result["profit_period_type"] = _PERIOD_TYPE_BY_QUARTER[period_quarter]
            result["profit_period_months"] = period_quarter * _MONTHS_PER_QUARTER

        return result

    app.tool()(get_stock_snapshot)
