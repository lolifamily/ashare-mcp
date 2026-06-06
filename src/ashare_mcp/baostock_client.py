"""Unified baostock client with session management and data normalization."""

from __future__ import annotations

import datetime as dt
import math
import re
import threading
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from typing import TYPE_CHECKING, Literal

import baostock as bs
import pandas as pd
from dateutil.tz import gettz

from ashare_mcp.errors import BaostockError, NoDataFoundError

if TYPE_CHECKING:
    from baostock.data.resultset import ResultData

_tz = gettz("Asia/Shanghai")
if _tz is None:
    _msg = "Asia/Shanghai timezone unavailable; is python-dateutil installed?"
    raise RuntimeError(_msg)
MARKET_TZ: dt.tzinfo = _tz

ZERO_THRESHOLD: float = 1e-9


def lookback_range(days: int, *, end: str | None = None) -> tuple[str, str]:
    """Return (start, end) 'YYYY-MM-DD' strings spanning `days` calendar days.

    end=None: anchor on today (Asia/Shanghai).
    end='YYYY-MM-DD': anchor on the user-supplied date.
    """
    if end is None:
        anchor = dt.datetime.now(tz=MARKET_TZ)
        end = anchor.strftime("%Y-%m-%d")
    else:
        anchor = dt.datetime.strptime(end, "%Y-%m-%d").replace(tzinfo=MARKET_TZ)
    start = (anchor - dt.timedelta(days=days)).strftime("%Y-%m-%d")
    return start, end


SEMICOLON_FIELDS: frozenset[str] = frozenset({
    "dividCashPsBeforeTax",
    "dividCashPsAfterTax",
})

NUMERIC_FIELDS: frozenset[str] = frozenset({
    # K-line OHLCV + valuation snapshots
    "open", "high", "low", "close", "preclose", "volume", "amount",
    "turn", "pctChg", "peTTM", "pbMRQ", "psTTM", "pcfNcfTTM",
    # profit_data
    "roeAvg", "npMargin", "gpMargin", "netProfit", "epsTTM",
    "MBRevenue", "totalShare", "liqaShare",
    # balance_data
    "currentRatio", "quickRatio", "cashRatio", "YOYLiability",
    "liabilityToAsset", "assetToEquity",
    # cash_flow_data
    "CAToAsset", "NCAToAsset", "tangibleAssetToAsset", "ebitToInterest",
    "CFOToOR", "CFOToNP", "CFOToGr",
    # growth_data
    "YOYEquity", "YOYAsset", "YOYNI", "YOYEPSBasic", "YOYPNI",
    # operation_data
    "NRTurnRatio", "NRTurnDays", "INVTurnRatio", "INVTurnDays",
    "CATurnRatio", "AssetTurnRatio",
    # dupont_data
    "dupontROE", "dupontAssetStoEquity", "dupontAssetTurn",
    "dupontPnitoni", "dupontNitogr", "dupontTaxBurden",
    "dupontIntburden", "dupontEbittogr",
    # dividend_data (semicolon fields are first summed to str, then coerced here)
    "dividCashPsBeforeTax", "dividCashPsAfterTax",
    "dividStocksPs", "dividCashStock", "dividReserveToStockPs",
    # performance_express_report
    "performanceExpressTotalAsset", "performanceExpressNetAsset",
    "performanceExpressEPSChgPct", "performanceExpressROEWa",
    "performanceExpressEPSDiluted", "performanceExpressGRYOY",
    "performanceExpressOPYOY",
    # forecast_report
    "profitForcastChgPctUp", "profitForcastChgPctDwn",
    # adjust_factor
    "foreAdjustFactor", "backAdjustFactor", "adjustFactor",
})

_SEMICOLON_RE: re.Pattern[str] = re.compile("[;" + chr(0xFF1B) + "]")

type BsQueryFn = Literal[
    "query_adjust_factor",
    "query_all_stock",
    "query_balance_data",
    "query_cash_flow_data",
    "query_deposit_rate_data",
    "query_dividend_data",
    "query_dupont_data",
    "query_forecast_report",
    "query_growth_data",
    "query_history_k_data_plus",
    "query_hs300_stocks",
    "query_loan_rate_data",
    "query_money_supply_data_month",
    "query_money_supply_data_year",
    "query_operation_data",
    "query_performance_express_report",
    "query_profit_data",
    "query_required_reserve_ratio_data",
    "query_stock_basic",
    "query_stock_industry",
    "query_sz50_stocks",
    "query_trade_dates",
    "query_zz500_stocks",
]

type BsParam = str | int | None
type Record = dict[str, object]


def _scalar(v: object) -> object:
    """Normalize a pandas scalar for JSON: NaN -> None, numpy scalar -> Python."""
    if v is None:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    return v


def as_float(v: object) -> float:
    """Cast a typed Record value to float. Raises on None / NaN / unconvertible."""
    if v is None:
        msg = "value is None"
        raise TypeError(msg)
    if isinstance(v, float):
        if math.isnan(v):
            msg = "value is NaN"
            raise ValueError(msg)
        return v
    if isinstance(v, (int, str)):
        return float(v)
    msg = f"cannot convert to float: {type(v).__name__}"
    raise TypeError(msg)


def df_to_records(df: pd.DataFrame) -> list[Record]:
    """Convert a typed DataFrame to a list of dicts. NaN -> None, types preserved."""
    return [{str(k): _scalar(v) for k, v in record.items()} for record in df.to_dict(orient="records")]


class Baostock:
    """Thin wrapper around baostock; all public methods serialize on self._lock.

    baostock holds a single process-global socket without internal locking, so
    concurrent send/recv would scramble its wire protocol. The lock is held for
    the full query lifecycle, including the streamed `while rs.next()` recv loop.
    """

    def __init__(self) -> None:
        """Initialize with no active session."""
        self._logged_in: bool = False
        self._lock = threading.Lock()

    def login(self) -> None:
        """Log in to the baostock server (idempotent, thread-safe)."""
        with self._lock:
            self._login_locked()

    def logout(self) -> None:
        """Log out from the baostock server (idempotent, thread-safe)."""
        with self._lock:
            self._logout_locked()

    def query(self, fn_name: BsQueryFn, **kwargs: BsParam) -> pd.DataFrame:
        """Call a baostock query function and return a normalized typed DataFrame.

        None-valued kwargs are dropped so callers can write `bs.query(..., date=date_or_none)`
        without having to build conditional kwargs dicts.
        """
        with self._lock:
            if not self._logged_in:
                self._login_locked()
            fn = getattr(bs, fn_name)
            clean: dict[str, str | int] = {k: v for k, v in kwargs.items() if v is not None}
            rs: ResultData = fn(**clean)
            if rs.error_code != "0":
                raise BaostockError(fn_name, dict(kwargs), rs.error_code, rs.error_msg)
            rows: list[list[str]] = []
            while rs.next():
                rows.append(rs.get_row_data())
            if not rows:
                raise NoDataFoundError(fn_name, dict(kwargs))
            df = pd.DataFrame(rows, columns=rs.fields)
            self._normalize(df)
            return df

    def query_one(self, fn_name: BsQueryFn, **kwargs: BsParam) -> Record:
        """Call a baostock query and return the first row as a typed Record."""
        row = self.query(fn_name, **kwargs).iloc[0]
        return {str(k): _scalar(v) for k, v in row.items()}

    def _login_locked(self) -> None:
        """Login implementation; caller must hold self._lock."""
        if self._logged_in:
            return
        buf = StringIO()
        with redirect_stdout(buf), redirect_stderr(buf):
            lg: ResultData = bs.login()
        if lg.error_code != "0":
            fn_name = "login"
            raise BaostockError(fn_name, {}, lg.error_code, lg.error_msg)
        self._logged_in = True

    def _logout_locked(self) -> None:
        """Logout implementation; caller must hold self._lock."""
        if not self._logged_in:
            return
        buf = StringIO()
        with redirect_stdout(buf), redirect_stderr(buf):
            bs.logout()
        self._logged_in = False

    def _normalize(self, df: pd.DataFrame) -> None:
        """In-place: sum semicolon multi-payouts and coerce numeric columns."""
        columns = frozenset(str(c) for c in df.columns)
        for col in SEMICOLON_FIELDS & columns:
            df[col] = df[col].apply(_split_semicolon_sum)
        for col in NUMERIC_FIELDS & columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")


def _split_semicolon_sum(value: object) -> str:
    """Split semicolon-separated numeric values and return their sum as a string."""
    if not isinstance(value, str) or not value.strip():
        return str(value)
    if not _SEMICOLON_RE.search(value):
        return value
    parts = _SEMICOLON_RE.split(value)
    total = 0.0
    for raw_part in parts:
        stripped = raw_part.strip()
        if not stripped:
            continue
        try:
            total += float(stripped)
        except ValueError:
            return value
    return str(total)
