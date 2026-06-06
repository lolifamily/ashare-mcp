"""Valuation analysis tools: PE/PB/PS, PEG, DDM, DCF, industry comparison, snapshot."""

from __future__ import annotations

import datetime as dt
import math
from typing import TYPE_CHECKING

import numpy as np

from ashare_mcp.baostock_client import MARKET_TZ, ZERO_THRESHOLD, Record, as_float, lookback_range
from ashare_mcp.derivations import derive_ocf, get_latest_close
from ashare_mcp.errors import BaostockError, NoDataFoundError

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

    from ashare_mcp.baostock_client import Baostock

VALUATION_FIELDS = "date,code,close,peTTM,pbMRQ,psTTM,pcfNcfTTM"

_MIN_DIVIDEND_YEARS = 2
_MIN_OCF_YEARS = 2
_DCF_HISTORY_YEARS = 5
_MAX_FORECAST_GROWTH = 0.20
_MIN_FORECAST_GROWTH = 0.01
_MAX_DCF_GROWTH = 0.15
_MIN_DCF_GROWTH = -0.05
_COMPARISON_LOOKBACK_DAYS = 7
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
    today_year: int, today_month: int, fallback_years: int,
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
    pairs.extend(
        (today_year - off, q)
        for off in range(1, fallback_years)
        for q in (4, 3, 2, 1)
    )
    return pairs
# Fraction (0..1) -> percent-basis number. Used both for percentile rank and for
# converting baostock YOY* ratios (e.g. 0.18) to the percent number (18) that
# the PEG convention expects.
_PERCENT_SCALE = 100


def _safe_float(val: object) -> float | None:
    """Convert a typed Record / Series scalar to float. None / NaN / unconvertible -> None."""
    if val is None:
        return None
    if isinstance(val, (int, float)):
        f = float(val)
    elif isinstance(val, str):
        try:
            f = float(val)
        except ValueError:
            return None
    else:
        return None
    return None if math.isnan(f) else f


def _clamp(value: float, lo: float, hi: float) -> tuple[float, str | None]:
    """Clamp value to [lo, hi]. Returns (clamped, reason_if_clamped)."""
    if value > hi:
        return hi, "exceeded_max"
    if value < lo:
        return lo, "below_min"
    return value, None


def register(app: FastMCP, bs: Baostock) -> None:
    """Register valuation tools with the MCP app."""
    _register_valuation_metrics(app, bs)
    _register_peg(app, bs)
    _register_ddm(app, bs)
    _register_dcf(app, bs)
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
        (the date of its last valid observation). Close prices and PE/PB/PS
        can be missing on different days (suspension, fresh IPO, data delay),
        so aligning them to a single "latest" date would either drop signals
        or silently mismatch. `period.last_trading_date` reports the last
        K-line bar in the window, separately from any individual metric's
        as_of date.

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
            code=code, fields=VALUATION_FIELDS,
            start_date=start_date, end_date=end_date,
            frequency="d", adjustflag="3",
        )

        metrics: dict[str, dict[str, float | str | None]] = {}
        for col in ["close", "peTTM", "pbMRQ", "psTTM", "pcfNcfTTM"]:
            vals = df[col].dropna()
            if vals.empty:
                metrics[col] = {"current": None, "as_of": None}
                continue
            current = float(vals.iloc[-1])
            as_of = str(df.loc[vals.index[-1], "date"])
            entry: dict[str, float | str | None] = {"current": current, "as_of": as_of}
            if col != "close":
                entry["mean"] = float(vals.mean())
                entry["min"] = float(vals.min())
                entry["max"] = float(vals.max())
                entry["percentile"] = float(np.mean(vals <= current)) * _PERCENT_SCALE
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
    def calculate_peg_ratio(code: str, year: str, quarter: int) -> dict[str, object]:
        """Calculate PEG = PE_TTM / G, where G is YoY net profit growth as a percent number.

        baostock's YOYNI is a ratio (e.g. 0.1858 = +18.58% YoY). The PEG convention
        expects G as the percent number (18.58), so we multiply by _PERCENT_SCALE.

        PEG is undefined and returns peg=None when:
          - PE is unavailable (suspended / no quote)
          - PE <= 0 (loss-making company; PE has no valuation meaning)
          - YoY growth <= 0 (PEG is a growth-stock metric; declining earnings
            yield a negative or nonsensical value that LLMs would misread as cheap)

        Args:
            code: Stock code.
            year: 4-digit year.
            quarter: 1-4.

        """
        start, end = lookback_range(_PEG_LOOKBACK_DAYS)
        price_data = bs.query(
            "query_history_k_data_plus",
            code=code, fields="date,close,peTTM",
            start_date=start, end_date=end,
            frequency="d", adjustflag="3",
        )
        pe_series = price_data["peTTM"].dropna()
        if pe_series.empty:
            return {
                "code": code, "pe_ttm": None, "yoyni": None, "peg": None,
                "reason": "no PE available (likely loss-making or recently suspended)",
            }
        pe = float(pe_series.iloc[-1])
        if pe <= 0:
            return {
                "code": code, "pe_ttm": pe, "yoyni": None, "peg": None,
                "reason": "PEG undefined for non-positive PE (loss-making company)",
            }

        growth = bs.query_one("query_growth_data", code=code, year=year, quarter=quarter)
        yoyni_raw = _safe_float(growth.get("YOYNI"))
        if yoyni_raw is None or yoyni_raw <= 0:
            return {
                "code": code, "pe_ttm": pe, "yoyni": yoyni_raw, "peg": None,
                "reason": "PEG undefined for non-positive growth",
            }

        # yoyni_raw > 0 here, so growth_pct > 0 — division is safe without ZERO_THRESHOLD guard.
        growth_pct = yoyni_raw * _PERCENT_SCALE
        return {
            "code": code,
            "pe_ttm": pe,
            "yoyni": yoyni_raw,
            "yoyni_pct": growth_pct,
            "peg": pe / growth_pct,
        }

    app.tool()(calculate_peg_ratio)


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
        Current price from latest K-line close.

        Args:
            code: Stock code.
            discount_rate: Required rate of return, e.g. 0.10.
            terminal_growth_rate: Perpetual growth rate, e.g. 0.025.
            years_back: Years of dividend history to use.
            forecast_years: Projection period.

        """
        if discount_rate <= terminal_growth_rate:
            msg = (
                f"discount_rate ({discount_rate}) must exceed terminal_growth_rate "
                f"({terminal_growth_rate}); Gordon model requires r > g"
            )
            raise ValueError(msg)

        current_year = dt.datetime.now(tz=MARKET_TZ).year
        annual_divs: dict[int, float] = {}
        for y in range(current_year - years_back, current_year + 1):
            try:
                df = bs.query("query_dividend_data", code=code, year=str(y), yearType="report")
                vals = df["dividCashPsBeforeTax"].dropna()
                if not vals.empty:
                    annual_divs[y] = float(vals.sum())
            except NoDataFoundError:
                continue

        if len(annual_divs) < _MIN_DIVIDEND_YEARS:
            return {"code": code, "error": "insufficient dividend history", "years_found": len(annual_divs)}

        years = sorted(annual_divs.keys())
        divs = [annual_divs[y] for y in years]

        # CAGR (geometric mean): consistent with DCF and correct for compound
        # growth. Arithmetic mean would massively overstate growth on a volatile
        # dividend series (e.g. [1,2,1,2,1] arithmetic = +25% but true CAGR = 0%).
        # divs[0] <= ZERO_THRESHOLD means the series starts at zero — CAGR is undefined; we
        # fall back to 0 growth, which is the conservative assumption for DDM.
        n_periods = len(divs) - 1
        cagr = (divs[-1] / divs[0]) ** (1 / n_periods) - 1 if divs[0] > ZERO_THRESHOLD and n_periods > 0 else 0.0
        forecast_growth, growth_clamped = _clamp(cagr, _MIN_FORECAST_GROWTH, _MAX_FORECAST_GROWTH)

        latest_div = divs[-1]
        pv_divs: list[dict[str, float]] = []
        for i in range(1, forecast_years + 1):
            future_div = latest_div * (1 + forecast_growth) ** i
            pv = future_div / (1 + discount_rate) ** i
            pv_divs.append({"year": float(years[-1] + i), "dividend": future_div, "pv": pv})

        terminal_div = latest_div * (1 + forecast_growth) ** forecast_years * (1 + terminal_growth_rate)
        terminal_value = terminal_div / (discount_rate - terminal_growth_rate)
        pv_terminal = terminal_value / (1 + discount_rate) ** forecast_years

        return {
            "code": code,
            "intrinsic_value_per_share": sum(d["pv"] for d in pv_divs) + pv_terminal,
            "current_price": get_latest_close(bs, code),
            "dividend_history": {str(y): d for y, d in zip(years, divs, strict=True)},
            "dividend_cagr": cagr,
            "forecast_growth": forecast_growth,
            "growth_clamped": growth_clamped,
            "growth_clamp_bounds": {
                "min": _MIN_FORECAST_GROWTH,
                "max": _MAX_FORECAST_GROWTH,
            },
            "projected_dividends": pv_divs,
            "pv_terminal": pv_terminal,
            "assumptions": {
                "discount_rate": discount_rate,
                "terminal_growth_rate": terminal_growth_rate,
                "forecast_years": forecast_years,
            },
        }

    app.tool()(calculate_ddm_valuation)


def _check_fcf_endpoints(
    code: str, years: list[int], fcfs: list[float],
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
        "hint": "try a lower capex_to_ocf_ratio or larger years_back",
    }


def _register_dcf(app: FastMCP, bs: Baostock) -> None:
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

        Historical window is fixed at _DCF_HISTORY_YEARS years; the FCF history
        gracefully falls back to as few as _MIN_OCF_YEARS years when reports are
        missing.

        Args:
            code: Stock code.
            discount_rate: WACC / discount rate, e.g. 0.10.
            terminal_growth_rate: Perpetual growth rate, e.g. 0.025.
            capex_to_ocf_ratio: Capex as fraction of OCF. No default — caller decides.
            net_debt: Interest-bearing debt minus cash & equivalents (CNY).
                Negative = net cash position. Required for per-share valuation.
            forecast_years: Projection period.

        """
        if discount_rate <= terminal_growth_rate:
            msg = (
                f"discount_rate ({discount_rate}) must exceed terminal_growth_rate "
                f"({terminal_growth_rate}); Gordon model requires r > g"
            )
            raise ValueError(msg)

        current_year = dt.datetime.now(tz=MARKET_TZ).year
        # Same range as DDM: include current year and let NoDataFoundError
        # naturally skip years whose Q4 report has not been published yet.
        # profit_history keeps each year's profit Record so totalShare lookup
        # reuses the last-year fetch without a second query.
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

        if len(ocf_history) < _MIN_OCF_YEARS:
            return {"code": code, "error": "insufficient OCF history", "years_found": len(ocf_history)}

        years = sorted(ocf_history.keys())
        ocfs = [ocf_history[y] for y in years]
        fcfs = [o * (1 - capex_to_ocf_ratio) for o in ocfs]

        endpoint_err = _check_fcf_endpoints(code, years, fcfs)
        if endpoint_err is not None:
            return endpoint_err

        ratio = fcfs[-1] / fcfs[0]
        n = len(fcfs) - 1
        cagr = ratio ** (1 / n) - 1 if ratio > ZERO_THRESHOLD and n > 0 else 0.0
        forecast_growth, growth_clamped = _clamp(cagr, _MIN_DCF_GROWTH, _MAX_DCF_GROWTH)

        projected: list[dict[str, float]] = []
        last_fcf = fcfs[-1]
        for i in range(1, forecast_years + 1):
            future_fcf = last_fcf * (1 + forecast_growth) ** i
            pv = future_fcf / (1 + discount_rate) ** i
            projected.append({"year": float(years[-1] + i), "fcf": future_fcf, "pv": pv})

        terminal_fcf = projected[-1]["fcf"] * (1 + terminal_growth_rate)
        terminal_value = terminal_fcf / (discount_rate - terminal_growth_rate)
        pv_terminal = terminal_value / (1 + discount_rate) ** forecast_years

        ev = sum(p["pv"] for p in projected) + pv_terminal

        result: dict[str, object] = {
            "code": code,
            "industry": str(bs.query_one("query_stock_industry", code=code)["industry"]),
            "current_price": get_latest_close(bs, code),
            "enterprise_value": ev,
            "intermediates": {
                "ocf_history": {str(y): v for y, v in zip(years, ocfs, strict=True)},
                "fcf_history": {str(y): v for y, v in zip(years, fcfs, strict=True)},
                "fcf_cagr": cagr,
                "forecast_growth": forecast_growth,
                "growth_clamped": growth_clamped,
                "growth_clamp_bounds": {
                    "min": _MIN_DCF_GROWTH,
                    "max": _MAX_DCF_GROWTH,
                },
            },
            "projected_fcf": projected,
            "pv_terminal": pv_terminal,
            "assumptions": {
                "discount_rate": discount_rate,
                "terminal_growth_rate": terminal_growth_rate,
                "capex_to_ocf_ratio": capex_to_ocf_ratio,
                "forecast_years": forecast_years,
            },
            "data_provenance": {
                "ocf": {"derived": True, "formula": "MBRevenue * CFOToOR", "approx_precision_pct": 2},
                "capex": {"user_provided": True, "note": "baostock provides no capex data"},
                "net_debt": {
                    "user_provided": True,
                    "note": "baostock provides no interest-bearing debt or cash breakdown",
                },
            },
        }

        if net_debt is not None:
            shares = as_float(profit_history[years[-1]]["totalShare"])
            equity_value = ev - net_debt
            result["net_debt"] = net_debt
            result["equity_value"] = equity_value
            result["total_shares"] = shares
            result["per_share_intrinsic_value"] = (
                equity_value / shares if shares > ZERO_THRESHOLD else 0.0
            )
        else:
            result["equity_note"] = (
                "per_share_intrinsic_value omitted: net_debt not provided. "
                "Pass net_debt (interest-bearing debt minus cash) to derive equity value."
            )

        return result

    app.tool()(calculate_dcf_valuation)


def _register_industry_comparison(app: FastMCP, bs: Baostock) -> None:
    def compare_industry_valuation(code: str, date: str | None = None) -> dict[str, object]:
        """Compare a stock's valuation (PE/PB/PS) against its industry peers.

        Behavior contract:
          - If `code` is not in the industry classification table for `date`,
            raises NoDataFoundError.
          - If `code` is in the classification table but its K-line is missing
            in the lookback window (suspension / fresh IPO / data delay),
            raises NoDataFoundError. The tool refuses to return a half-result
            with target=None, which would invite the caller to silently switch
            from "target vs peers" to "industry overview only".
          - Peers whose K-line is missing are listed in `peer_coverage.skipped`
            with the reason. Industry statistics are computed on the published
            subset; the caller can judge sample-bias risk from the skipped list.

        Args:
            code: Target stock code.
            date: Comparison date 'YYYY-MM-DD'. Defaults to latest.

        """
        # Single industry query: target's industry is filtered out of the full
        # snapshot, so target_industry and peers share the same date semantics
        # (previously they didn't — target was latest, peers respected `date`).
        all_ind = bs.query("query_stock_industry", date=date)
        target_row = all_ind[all_ind["code"] == code]
        if target_row.empty:
            fn = "compare_industry_valuation"
            raise NoDataFoundError(fn, {"code": code, "date": date or "latest"})
        target_industry = str(target_row.iloc[0]["industry"])
        peers = all_ind[all_ind["industry"] == target_industry]

        start_date, end_date = lookback_range(_COMPARISON_LOOKBACK_DAYS, end=date)

        valuations: list[dict[str, object]] = []
        skipped: list[dict[str, str]] = []
        for _, row in peers.iterrows():
            sc = str(row["code"])
            try:
                vdf = bs.query(
                    "query_history_k_data_plus",
                    code=sc, fields=VALUATION_FIELDS,
                    start_date=start_date, end_date=end_date,
                    frequency="d", adjustflag="3",
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
            valuations.append({
                "code": sc,
                "code_name": str(row.get("code_name", sc)),
                "close": _safe_float(latest["close"]),
                "peTTM": _safe_float(latest["peTTM"]),
                "pbMRQ": _safe_float(latest["pbMRQ"]),
                "psTTM": _safe_float(latest["psTTM"]),
                "is_target": sc == code,
            })

        # Target self-check: classification table said target exists, but its
        # K-line query may still have failed (suspension / IPO timing / etc).
        # Refuse to return a meaningful-looking result without the target.
        target_records = [v for v in valuations if v.get("is_target")]
        if not target_records:
            fn = "compare_industry_valuation"
            raise NoDataFoundError(
                fn,
                {
                    "code": code,
                    "industry": target_industry,
                    "issue": "target has no K-line in lookback window",
                    "peers_published": len(valuations),
                    "peers_skipped": len(skipped),
                },
            )

        stats: dict[str, dict[str, float]] = {}
        for metric in ["peTTM", "pbMRQ", "psTTM"]:
            vals_list = [f for v in valuations if (f := _safe_float(v.get(metric))) is not None]
            if vals_list:
                arr = np.array(vals_list)
                stats[metric] = {
                    "mean": float(np.mean(arr)),
                    "median": float(np.median(arr)),
                    "min": float(np.min(arr)),
                    "max": float(np.max(arr)),
                }

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


def _register_snapshot(app: FastMCP, bs: Baostock) -> None:
    def get_stock_snapshot(code: str) -> dict[str, object]:
        """One-call snapshot: latest price, valuation, industry, total shares, net profit.

        Combines data from K-line, stock_industry, and profit_data.

        Profit fields (`net_profit`, `revenue`) come from the most recently
        published report, scanned in reverse (year, quarter) order — so they
        may be Q1, H1, 9M, or FY depending on the calendar. Always inspect
        `profit_period_type` ('Q1' / 'H1' / '9M' / 'FY') and
        `profit_period_months` (3 / 6 / 9 / 12) before comparing across
        companies: baostock reports these as cumulative-from-year-start, so
        Q1's net_profit is 3 months and is NOT directly comparable to FY's
        12 months. To annualize Q1, multiply by 4; to annualize H1, by 2.

        Args:
            code: Stock code.

        """
        # Single K-line query: VALUATION_FIELDS already contains `close`, so a
        # separate get_latest_close() call would just duplicate the network round-trip.
        start, end = lookback_range(_SNAPSHOT_LOOKBACK_DAYS)
        kdf = bs.query(
            "query_history_k_data_plus",
            code=code, fields=VALUATION_FIELDS,
            start_date=start, end_date=end,
            frequency="d", adjustflag="3",
        )
        latest_k = kdf.iloc[-1]
        price = _safe_float(latest_k["close"])

        ind = bs.query_one("query_stock_industry", code=code)
        basic = bs.query_one("query_stock_basic", code=code)

        # Scan (year, quarter) in reverse chronological order, pruned to
        # quarters whose disclosure window has opened — the newest
        # published report wins regardless of which quarter it happens
        # to be. baostock returns cumulative YTD values; callers must
        # read profit_period_type before comparing across companies.
        profit: Record | None = None
        period_quarter: int | None = None
        now = dt.datetime.now(tz=MARKET_TZ)
        for year, quarter in _profit_candidates(now.year, now.month, _PROFIT_FALLBACK_YEARS):
            try:
                profit = bs.query_one(
                    "query_profit_data", code=code, year=str(year), quarter=quarter,
                )
            except NoDataFoundError:
                continue
            period_quarter = quarter
            break

        result: dict[str, object] = {
            "code": code,
            "code_name": ind.get("code_name") or basic.get("code_name"),
            "industry": ind["industry"],
            "ipo_date": basic.get("ipoDate"),
            "close": price,
            "peTTM": _safe_float(latest_k["peTTM"]),
            "pbMRQ": _safe_float(latest_k["pbMRQ"]),
            "psTTM": _safe_float(latest_k["psTTM"]),
        }
        if profit and period_quarter is not None:
            result["total_shares"] = _safe_float(profit["totalShare"])
            result["net_profit"] = _safe_float(profit["netProfit"])
            result["revenue"] = _safe_float(profit["MBRevenue"])
            result["profit_period"] = profit.get("statDate")
            result["profit_period_type"] = _PERIOD_TYPE_BY_QUARTER[period_quarter]
            result["profit_period_months"] = period_quarter * _MONTHS_PER_QUARTER

        return result

    app.tool()(get_stock_snapshot)
