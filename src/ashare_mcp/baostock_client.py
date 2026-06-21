"""Unified baostock client with session management and data normalization."""

from __future__ import annotations

import re
import threading
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from typing import TYPE_CHECKING, Literal

import baostock as bs
import baostock.common.context as _bs_context
import pandas as pd

from ashare_mcp.errors import BaostockError, NoDataFoundError
from ashare_mcp.utils import Record, scalar

if TYPE_CHECKING:
    from baostock.data.resultset import ResultData

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
    # money_supply (month: m{0,1,2}{Month,YOY,ChainRelative})
    "m0Month", "m0YOY", "m0ChainRelative",
    "m1Month", "m1YOY", "m1ChainRelative",
    "m2Month", "m2YOY", "m2ChainRelative",
    # money_supply (year: m{0,1,2}{Year,YearYOY})
    "m0Year", "m0YearYOY",
    "m1Year", "m1YearYOY",
    "m2Year", "m2YearYOY",
    # deposit_rate (statYear/statMonth/pubDate stay as str — labels & dates)
    "demandDepositRate",
    "fixedDepositRate3Month", "fixedDepositRate6Month",
    "fixedDepositRate1Year", "fixedDepositRate2Year",
    "fixedDepositRate3Year", "fixedDepositRate5Year",
    "installmentFixedDepositRate1Year",
    "installmentFixedDepositRate3Year",
    "installmentFixedDepositRate5Year",
    # loan_rate (baostock literally spells it "mortgate" — keep the typo, it's the wire name)
    "loanRate6Month", "loanRate6MonthTo1Year",
    "loanRate1YearTo3Year", "loanRate3YearTo5Year", "loanRateAbove5Year",
    "mortgateRateBelow5Year", "mortgateRateAbove5Year",
    # required_reserve_ratio
    "bigInstitutionsRatioPre", "bigInstitutionsRatioAfter",
    "mediumInstitutionsRatioPre", "mediumInstitutionsRatioAfter",
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

# baostock error codes that warrant a re-login + single retry: the server-side
# session lapsed (10001001: expired, or evicted by a concurrent login on the same
# account) or a transient network fault dropped the socket (10002xxx). Everything
# else -- parameter errors (10004xxx), bad credentials (10001002), version expiry
# (10001004) -- is raised as-is: retrying wastes a round-trip and can mask the real
# error behind a spurious login failure.
_RETRY_ERROR_CODES: frozenset[str] = frozenset({
    "10001001",  # not logged in: session expired or evicted by a concurrent login
    "10002001", "10002002", "10002003", "10002004",  # network: connection lost
    "10002005", "10002006", "10002007", "10002008",  # network: send/recv fail or timeout
})

# baostock validates some params client-side by printing a message and returning
# None instead of a ResultData (e.g. a malformed start_date). Sentinel error code
# so the caller gets a BaostockError, not 'NoneType has no attribute error_code'.
_NULL_RESULT_CODE = "NULL_RESULT"

# baostock treats an empty `code` as "all securities" (query_stock_basic returns
# the whole market), so query_one would silently hand back a random first row.
# Reject blank codes loudly; use code=None for queries that legitimately mean "all".
_EMPTY_CODE = "EMPTY_CODE"

# baostock's send_msg() does `while True: socket.recv(8192)` on a socket that
# has NO timeout set (baostock/util/socketutil.py creates it with the default
# blocking mode and never calls settimeout). Certain malformed params -- observed
# with query_trade_dates + start_date='2026-13-99' -- cause the server to never
# send the closing `<![CDATA[]]>\n` sentinel, so the recv loop blocks forever.
# Because our query() holds self._lock for the full call lifetime, that pins the
# entire MCP server.
#
# We don't need a wrapper thread to fix this: baostock exposes the socket as a
# module attribute (`baostock.common.context.default_socket`), so we just set an
# OS-level recv/send timeout on it after each login. On timeout, socket.recv
# raises socket.timeout, baostock's send_msg() catches it and returns None, and
# the per-API wrapper translates that to BSERR_RECVSOCK_FAIL ('10002007') --
# already in _RETRY_ERROR_CODES. The single retry triggers a fresh bs.login()
# which builds a brand new socket, so no half-broken state leaks into the next
# call. 30s sits well above the slowest observed normal call (~10s for
# query_stock_industry full table) and below the MCP client's per-tool
# deadline.
_QUERY_TIMEOUT_S = 30.0


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

        baostock's server-side session can lapse out from under us (idle timeout, or a
        concurrent login on the same account evicting ours), and the optimistic
        self._logged_in flag can't see that -- left alone, one lapse fails every later
        call forever. On a session/network error code -- whether surfaced as the query
        result's error_code or raised by _login_locked itself when bs.login() flakes --
        we drop the flag and retry once, forcing a fresh login. Parameter errors
        (10004xxx) are raised without retry. Retry is globally capped at 1: the second
        _run_locked call propagates any failure as-is, so a login failure inside it is
        NOT caught again -- this prevents nested login retries from compounding.
        """
        code = kwargs.get("code")
        if isinstance(code, str) and not code.strip():
            msg = "code must be non-empty; pass code=None for an 'all securities' query"
            raise BaostockError(fn_name, dict(kwargs), _EMPTY_CODE, msg)
        with self._lock:
            clean: dict[str, str | int] = {k: v for k, v in kwargs.items() if v is not None}
            try:
                rs = self._run_locked(fn_name, clean)
                retry = rs is not None and rs.error_code in _RETRY_ERROR_CODES
            except BaostockError as e:
                # _login_locked failed inside _run_locked. If the failure is a retryable
                # network/session code, fall through to the same retry path query already
                # uses for rs.error_code. Non-retryable codes (bad credentials, etc.) raise.
                if e.code not in _RETRY_ERROR_CODES:
                    raise
                rs, retry = None, True
            if retry:
                self._logged_in = False  # session is dead; _run_locked re-logs in on retry
                rs = self._run_locked(fn_name, clean)
            if rs is None:
                msg = "baostock returned None (malformed params?)"
                raise BaostockError(fn_name, dict(kwargs), _NULL_RESULT_CODE, msg)
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

    def _run_locked(self, fn_name: BsQueryFn, clean: dict[str, str | int]) -> ResultData | None:
        """Log in if needed, then run the baostock query fn. Caller must hold self._lock.

        Returns baostock's raw result, including the None it yields for some
        client-side param validation failures; query()'s guard turns that into a
        BaostockError.

        Suppresses stdout/stderr around the call: on client-side validation
        failure (bad date / quarter / code format / reversed range) baostock
        writes CN error messages to stdout via `print()`, which corrupts the
        JSON-RPC frames the MCP stdio transport multiplexes through the same
        fd. Same defensive pattern as _login_locked / _logout_locked.
        """
        if not self._logged_in:
            self._login_locked()
        fn = getattr(bs, fn_name)
        buf = StringIO()
        with redirect_stdout(buf), redirect_stderr(buf):
            return fn(**clean)  # may be None for malformed params

    def query_one(self, fn_name: BsQueryFn, **kwargs: BsParam) -> Record:
        """Call a baostock query and return the first row as a typed Record."""
        row = self.query(fn_name, **kwargs).iloc[0]
        return {str(k): scalar(v) for k, v in row.items()}

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
        # bs.login() built a fresh socket inside SocketUtil and stashed it on
        # the `default_socket` module attribute. Inject the recv/send deadline
        # here so every subsequent send_msg() inherits it -- see _QUERY_TIMEOUT_S
        # comment for the rationale. The attribute is set by SocketUtil.connect();
        # if a future baostock refactor renames or removes it, getattr returns
        # None and we skip the call rather than crash -- the hang behavior
        # silently returns, which a future test should catch.
        sock = getattr(_bs_context, "default_socket", None)
        if sock is not None:
            sock.settimeout(_QUERY_TIMEOUT_S)
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
