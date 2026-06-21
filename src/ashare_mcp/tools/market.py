"""Stock market data tools: K-line, basic info, dividends, calendar."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ashare_mcp.errors import NoDataFoundError
from ashare_mcp.utils import Record, df_to_records, lookback_range, safe_float

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

    from ashare_mcp.baostock_client import Baostock

K_FIELDS_DAILY = (
    "date,code,open,high,low,close,preclose,volume,amount,"
    "adjustflag,turn,tradestatus,pctChg,peTTM,pbMRQ,psTTM,pcfNcfTTM,isST"
)
K_FIELDS_WEEKLY = "date,code,open,high,low,close,volume,amount,adjustflag,turn,pctChg"
K_FIELDS_MINUTE = "date,time,code,open,high,low,close,volume,amount,adjustflag"

_K_FIELDS_BY_FREQ: dict[str, str] = {
    "d": K_FIELDS_DAILY,
    "w": K_FIELDS_WEEKLY,
    "m": K_FIELDS_WEEKLY,
    "5": K_FIELDS_MINUTE,
    "15": K_FIELDS_MINUTE,
    "30": K_FIELDS_MINUTE,
    "60": K_FIELDS_MINUTE,
}

# 15 = CN Spring Festival max closure (9 days) + buffer; guarantees the CSI 300
# probe finds at least one published bar regardless of calendar position.
_LATEST_TRADE_LOOKBACK_DAYS = 15
# CSI 300 is the de-facto liquidity proxy for A-shares: if it has no bar for
# date D, no individual stock will either. Used to detect "data published" state.
_LATEST_DATE_PROBE_CODE = "sh.000300"


def _latest_trading_date(bs: Baostock) -> str:
    """Return the most recent trading date with a published CSI 300 bar ('YYYY-MM-DD').

    Probes CSI 300 over a short lookback window and returns its latest bar date —
    the safe anchor for queries whose data source treats an absent date as "today"
    (which yields nothing pre-market or on non-trading days).
    """
    start, end = lookback_range(_LATEST_TRADE_LOOKBACK_DAYS)
    df = bs.query(
        "query_history_k_data_plus",
        code=_LATEST_DATE_PROBE_CODE,
        fields="date",
        start_date=start,
        end_date=end,
        frequency="d",
        adjustflag="3",
    )
    return str(df["date"].max())


def register(app: FastMCP, bs: Baostock) -> None:
    """Register market data tools with the MCP app."""

    def get_historical_k_data(
        code: str,
        start_date: str,
        end_date: str,
        frequency: str = "d",
        adjust_flag: str = "3",
        fields: str | None = None,
    ) -> list[Record]:
        """Fetch historical K-line (OHLCV + valuation) data for a Chinese A-share stock.

        Args:
            code: Stock code in baostock format, e.g. 'sh.600000', 'sz.000001'.
            start_date: Start date 'YYYY-MM-DD'.
            end_date: End date 'YYYY-MM-DD'.
            frequency: 'd' daily, 'w' weekly, 'm' monthly, '5'/'15'/'30'/'60' minutes.
            adjust_flag: '1' backward/后复权, '2' forward/前复权, '3' unadjusted/不复权 (default '3').
            fields: Comma-separated field list. Defaults to all standard fields.

        """
        default = _K_FIELDS_BY_FREQ.get(frequency, K_FIELDS_DAILY)
        df = bs.query(
            "query_history_k_data_plus",
            code=code,
            fields=fields or default,
            start_date=start_date,
            end_date=end_date,
            frequency=frequency,
            adjustflag=adjust_flag,
        )
        return df_to_records(df)

    app.tool()(get_historical_k_data)

    def get_stock_basic_info(code: str) -> Record:
        """Fetch basic info: code, code_name, ipoDate, outDate, type, status.

        Args:
            code: Stock code, e.g. 'sh.600519'.

        """
        return bs.query_one("query_stock_basic", code=code)

    app.tool()(get_stock_basic_info)

    def get_dividend_data(code: str, year: str, year_type: str = "report") -> dict[str, object]:
        """Fetch a year's dividends: annual cash total plus every payout's detail.

        A year with several distributions has one entry per payout in `payouts`
        (full detail). `annual_cash_per_share_pretax` is their summed
        dividCashPsBeforeTax — the yearly total, so callers needn't add across
        rows. Within a single payout, semicolon-separated multi-values in the
        cash-dividend fields are already summed by the client.

        annual_cash_per_share_pretax is None when the year had distributions but
        no cash (pure stock dividend / capital reserve conversion); read each
        payout's dividStocksPs / dividReserveToStockPs for those.

        Args:
            code: Stock code.
            year: 4-digit year, e.g. '2023'.
            year_type: 'report' (announcement year) or 'operate' (ex-dividend year).

        Returns:
            {code, year, annual_cash_per_share_pretax, payout_count, payouts}.

        """
        payouts = df_to_records(bs.query("query_dividend_data", code=code, year=year, yearType=year_type))
        cash = [f for p in payouts if (f := safe_float(p.get("dividCashPsBeforeTax"))) is not None]
        # None, not 0.0: A-share firms either pay cash or don't, so 0.0 isn't a
        # real datapoint — collapsing the no-cash year into the same value as
        # "paid 0 CNY/share" would hide the stock-only payouts in `payouts`.
        # round to 10dp: payouts come pre-rounded via scalar() but sum() reintroduces
        # IEEE 754 noise (e.g. 27.673 + 23.957 = 51.629999999999995); match scalar()'s
        # precision so the aggregate doesn't look spurious next to the components.
        return {
            "code": code,
            "year": year,
            "annual_cash_per_share_pretax": round(sum(cash), 10) if cash else None,
            "payout_count": len(payouts),
            "payouts": payouts,
        }

    app.tool()(get_dividend_data)

    def get_adjust_factor_data(code: str, start_date: str, end_date: str) -> list[Record]:
        """Fetch adjustment factor data for calculating adjusted prices.

        Factors exist only on ex-div / ex-rights dates; a window with none
        returns an empty list.

        Args:
            code: Stock code.
            start_date: 'YYYY-MM-DD'.
            end_date: 'YYYY-MM-DD'.

        """
        try:
            df = bs.query("query_adjust_factor", code=code, start_date=start_date, end_date=end_date)
        except NoDataFoundError:
            return []
        return df_to_records(df)

    app.tool()(get_adjust_factor_data)

    def get_trade_dates(
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> list[Record]:
        """Fetch trading calendar. Each row has calendar_date and is_trading_day ('1'/'0').

        Args:
            start_date: Optional 'YYYY-MM-DD'.
            end_date: Optional 'YYYY-MM-DD'.

        """
        return df_to_records(bs.query("query_trade_dates", start_date=start_date, end_date=end_date))

    app.tool()(get_trade_dates)

    def get_latest_trading_date() -> str:
        """Most recent trading date with K-line data published. Returns 'YYYY-MM-DD'.

        Probes CSI 300 (sh.000300) directly and returns the date of its latest
        available daily bar. This is the safe anchor date for follow-up K-line
        queries — guaranteed to have data, regardless of wall-clock time. On a
        trading day before market close, today's bar is not yet published, so
        this returns the previous trading date.
        """
        return _latest_trading_date(bs)

    app.tool()(get_latest_trading_date)

    def get_all_stock(query: str, date: str | None = None) -> list[Record]:
        """Search stocks by name. Returns matching code, code_name, and tradeStatus.

        Args:
            query: Search keyword matched against code_name (substring, case-insensitive).
                An empty string matches every row and returns the full market
                (~5000+ stocks, ~1 MB).
            date: Date 'YYYY-MM-DD'. Defaults to the latest trading date.

        """
        if date is None:
            date = _latest_trading_date(bs)
        df = bs.query("query_all_stock", day=date)
        # regex=False: doc promises substring match. The pandas default regex=True
        # would interpret "ST*" as a quantifier (match anything containing S
        # zero-or-more times -> matches ~every code_name), and trip on unbalanced
        # metachars like "(银行" / "[xyz" with an opaque re.error -- both seen in
        # the bug hunt. Plain substring is what users mean.
        mask = df["code_name"].str.contains(query, case=False, na=False, regex=False)
        return df_to_records(df[mask])

    app.tool()(get_all_stock)
