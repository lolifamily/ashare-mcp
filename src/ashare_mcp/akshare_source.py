"""Optional akshare adapter for EastMoney financial statements."""

from __future__ import annotations

import datetime as dt
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from typing import TYPE_CHECKING

import pandas as pd

from ashare_mcp.errors import AkshareError
from ashare_mcp.utils import MARKET_TZ, safe_float

if TYPE_CHECKING:
    import types
    from typing import TypeGuard

try:
    import akshare as _ak
except ImportError:
    _ak = None

AKSHARE_AVAILABLE: bool = _ak is not None

_REPORT_DATE_COL = "REPORT_DATE"
_OCF_COL = "NETCASH_OPERATE"
_CAPEX_COL = "CONSTRUCT_LONG_ASSET"
# Net debt components. The strict definition is short-term debt + current portion
# of long-term debt + long-term debt - cash, but EastMoney only exposes the
# "current portion of LTD" rolled up into NONCURRENT_LIAB_1YEAR — a CN-GAAP
# aggregate that also contains lease-liability and long-payable current portions
# (not interest-bearing). Including the whole bucket overstates net debt for
# lease-heavy / zero-leverage firms; excluding it would understate it for
# leveraged firms whose bucket is mostly LTD current portion. The upper-bound
# choice matches the fail-loud policy elsewhere; net_debt()'s docstring tells
# downstream consumers it's an upper bound.
_NET_DEBT_COMPONENTS: dict[str, int] = {
    "SHORT_LOAN": 1,
    "LONG_LOAN": 1,
    "BOND_PAYABLE": 1,
    "NONCURRENT_LIAB_1YEAR": 1,
    "MONETARYFUNDS": -1,
}
# A-share fiscal years equal the calendar year by law, so annual reports always
# carry REPORT_DATE month-day 12-31 — a structural signal, unlike the
# REPORT_DATE_NAME text which is UI-layer and can change.
_ANNUAL_REPORT_MONTH_DAY = (12, 31)


def _is_real_date(v: object) -> TypeGuard[dt.date]:
    """Return True only for a genuine date value.

    pd.NaT is a dt.date subclass, so a bare isinstance(v, dt.date) lets it
    through — yet comparing NaT against a real date raises TypeError. Exclude it
    explicitly, the same NaT guard scalar() applies at the JSON boundary.
    """
    return isinstance(v, dt.date) and not pd.isna(v)


def _to_em(code: str) -> str:
    """Convert baostock code to EastMoney symbol: 'sh.600519' -> 'SH600519'."""
    return code.replace(".", "").upper()


def _call(ak: types.ModuleType, fn_name: str, code: str) -> pd.DataFrame:
    """Call an akshare function; wrap any exception in AkshareError.

    Suppresses stdout/stderr because akshare emits tqdm progress bars to stderr
    (~1KB/call) and may print to stdout — both break MCP's stdio JSON-RPC framing.
    Same defensive pattern as Baostock._login_locked / _logout_locked.
    """
    sym = _to_em(code)
    buf = StringIO()
    try:
        with redirect_stdout(buf), redirect_stderr(buf):
            return getattr(ak, fn_name)(symbol=sym)
    except Exception as e:
        cause, no_data = _explain(e)
        raise AkshareError(fn_name, {"symbol": sym}, cause, no_data=no_data) from e


def _explain(e: Exception) -> tuple[Exception, bool]:
    """Classify an akshare exception; returns (cause, no_data).

    When EastMoney's datacenter endpoint serves no data for a (symbol, statement)
    pair it returns {"result": null}; akshare does not guard this and crashes deep
    inside its DataFrame construction with `TypeError: 'NoneType' object is not
    subscriptable`. Detect that and report an empty payload (no_data=True); every
    other exception passes through unchanged with no_data=False.

    Why the payload is empty (financial-sector template, rate-limit, a symbol with
    no such report) is not knowable here, so we don't guess it in the message --
    callers branch on the no_data flag, not on prose.
    """
    if isinstance(e, TypeError) and "subscriptable" in str(e):
        return ValueError("akshare returned no data"), True
    return e, False


class AkshareSource:
    """EastMoney financial statement adapter via akshare."""

    def __init__(self) -> None:
        """Initialize; raises RuntimeError if akshare was not importable."""
        if _ak is None:
            msg = "akshare is not installed"
            raise RuntimeError(msg)
        self._ak: types.ModuleType = _ak

    # The `*_by_report_delisted_em` functions hit EastMoney's datacenter API
    # (ps=200, one bulk fetch) instead of the `*_by_report_em` paginated endpoint
    # (5 report-dates per request -> ~22 requests). "delisted" is a misnomer: the
    # datacenter API works for any listing status, returns the same fields, and is
    # verified to yield identical values ~10x faster. Kept here so a single file
    # changes if akshare ever guards these against live stocks.
    def balance_sheet(self, code: str) -> pd.DataFrame:
        """Full balance sheet (all periods, all columns)."""
        return _call(self._ak, "stock_balance_sheet_by_report_delisted_em", code)

    def income_statement(self, code: str) -> pd.DataFrame:
        """Full income statement (all periods, all columns)."""
        return _call(self._ak, "stock_profit_sheet_by_report_delisted_em", code)

    def cash_flow(self, code: str) -> pd.DataFrame:
        """Full cash flow statement (all periods, all columns)."""
        return _call(self._ak, "stock_cash_flow_sheet_by_report_delisted_em", code)

    def ocf_and_capex_history(self, code: str, years_back: int) -> dict[int, tuple[float, float]]:
        """Per-year (OCF, Capex) from cash flow statement.

        Returns {year: (ocf, capex)} for annual reports with year in
        [current_year - years_back, current_year] — inclusive lower bound,
        matching _gather_fcf_baostock so both DCF paths use the same window.
        Capex is returned as a positive number (original is negative outflow).

        Annual reports are matched by REPORT_DATE month-day == 12-31 (A-share
        fiscal years equal the calendar year by law) — a structural signal,
        unlike the REPORT_DATE_NAME text which is UI-layer and can change.
        akshare casts REPORT_DATE to datetime.date via pd.to_datetime().dt.date;
        isinstance(date) also accepts a datetime/Timestamp subclass, so dropping
        that cast upstream stays compatible, while a non-date value (column
        renamed/removed) means the schema shifted — we raise rather than
        silently fall back to the lower-precision baostock estimate.
        """
        df = self.cash_flow(code)
        current_year = dt.datetime.now(tz=MARKET_TZ).year
        oldest_year = current_year - years_back
        result: dict[int, tuple[float, float]] = {}
        parsed_any_date = False
        for _, row in df.iterrows():
            d = row.get(_REPORT_DATE_COL)
            if not _is_real_date(d):
                continue
            parsed_any_date = True
            if d.year < oldest_year or (d.month, d.day) != _ANNUAL_REPORT_MONTH_DAY:
                continue
            ocf = safe_float(row.get(_OCF_COL))
            capex_raw = safe_float(row.get(_CAPEX_COL))
            if ocf is None or capex_raw is None:
                continue
            result.setdefault(d.year, (ocf, abs(capex_raw)))
        if not df.empty and not parsed_any_date:
            fn = "ocf_and_capex_history"
            msg = f"{_REPORT_DATE_COL} not date-typed; akshare schema may have changed"
            raise AkshareError(fn, {"code": code}, ValueError(msg))
        return result

    def net_debt(self, code: str) -> dict[str, object]:
        """Compute net debt from the latest balance sheet (conservative upper bound).

        net_debt = SHORT_LOAN + LONG_LOAN + BOND_PAYABLE + NONCURRENT_LIAB_1YEAR - MONETARYFUNDS

        NONCURRENT_LIAB_1YEAR is a CN-GAAP aggregate of all non-current liabilities
        due within 12 months: it contains long-term-debt current portion
        (interest-bearing) plus lease-liability and long-payable current portions
        (not interest-bearing), with no breakdown exposed by EastMoney. The
        returned net_debt is an upper bound — for lease-heavy or zero-leverage
        firms it is overstated by the non-interest portion of this bucket.

        A missing component COLUMN means the EastMoney schema drifted (field
        renamed/removed) — raise rather than silently treat it as 0 and understate
        the debt, the same fail-loud policy as ocf_and_capex_history. A column that
        exists but is NaN on the latest row is a legitimate business zero (e.g. no
        bonds issued) and contributes 0.
        """
        df = self.balance_sheet(code)
        if df.empty:
            fn = "net_debt"
            raise AkshareError(fn, {"code": code}, ValueError("empty balance sheet"))
        missing = [col for col in _NET_DEBT_COMPONENTS if col not in df.columns]
        if missing:
            fn = "net_debt"
            msg = f"balance sheet missing columns {missing}; akshare schema may have changed"
            raise AkshareError(fn, {"code": code}, ValueError(msg))
        # Pick the latest report by REPORT_DATE, not row position: akshare's
        # "iloc[0] = newest" is a documented contract, but the date column is the
        # real source of truth, so a future ordering change can't silently feed us
        # a stale balance sheet. Same dt.date structural signal as ocf_and_capex_history.
        if _REPORT_DATE_COL not in df.columns:
            fn = "net_debt"
            msg = f"{_REPORT_DATE_COL} column missing; akshare schema may have changed"
            raise AkshareError(fn, {"code": code}, ValueError(msg))
        # enumerate (not .items()) so we index back with .iloc[pos], which stays a
        # well-typed Series; .loc[label] returns an untyped scalar/Series under the stubs.
        dated = [(d, pos) for pos, d in enumerate(df[_REPORT_DATE_COL]) if _is_real_date(d)]
        if not dated:
            fn = "net_debt"
            msg = f"{_REPORT_DATE_COL} not date-typed; akshare schema may have changed"
            raise AkshareError(fn, {"code": code}, ValueError(msg))
        row = df.iloc[max(dated, key=lambda t: t[0])[1]]
        components: dict[str, float | None] = {}
        total = 0.0
        valid = 0
        for col, sign in _NET_DEBT_COMPONENTS.items():
            val = safe_float(row.get(col))
            components[col] = val
            if val is not None:
                total += sign * val
                valid += 1
        if valid == 0:
            fn = "net_debt"
            msg = "all net-debt component columns are empty on the latest row"
            raise AkshareError(fn, {"code": code}, ValueError(msg))
        report_date = row.get(_REPORT_DATE_COL)
        result: dict[str, object] = {
            "net_debt": total,
            "components": components,
            "report_date": str(report_date) if report_date is not None else None,
        }
        # NaN components were counted as 0 (a true business zero, e.g. no bonds).
        # But the schema can't tell a real zero from missing data, so surface which
        # ones were null: a missing debt item understates net_debt, missing cash
        # overstates it. components already shows them; this makes the risk explicit.
        null_components = [col for col, val in components.items() if val is None]
        if null_components:
            result["warning"] = (
                f"components {null_components} are null on the latest report and counted as 0; "
                "if that is missing data rather than a true zero, net_debt is understated "
                "(missing debt) or overstated (missing cash)"
            )
        return result
