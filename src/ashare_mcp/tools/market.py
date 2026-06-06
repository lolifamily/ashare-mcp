"""Stock market data tools: K-line, basic info, dividends, calendar."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ashare_mcp.baostock_client import Record, df_to_records, lookback_range

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

    from ashare_mcp.baostock_client import Baostock

K_FIELDS = (
    "date,code,open,high,low,close,preclose,volume,amount,"
    "adjustflag,turn,tradestatus,pctChg,peTTM,pbMRQ,psTTM,pcfNcfTTM,isST"
)

# 15 = CN Spring Festival max closure (9 days) + buffer; guarantees the CSI 300
# probe finds at least one published bar regardless of calendar position.
_LATEST_TRADE_LOOKBACK_DAYS = 15
# CSI 300 is the de-facto liquidity proxy for A-shares: if it has no bar for
# date D, no individual stock will either. Used to detect "data published" state.
_LATEST_DATE_PROBE_CODE = "sh.000300"


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
            adjust_flag: '1' forward, '2' backward, '3' unadjusted.
            fields: Comma-separated field list. Defaults to all standard fields.

        """
        df = bs.query(
            "query_history_k_data_plus",
            code=code,
            fields=fields or K_FIELDS,
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

    def get_dividend_data(code: str, year: str, year_type: str = "report") -> list[Record]:
        """Fetch dividend data. Multi-payout values are auto-summed.

        Args:
            code: Stock code.
            year: 4-digit year, e.g. '2023'.
            year_type: 'report' (announcement year) or 'operate' (ex-dividend year).

        """
        return df_to_records(bs.query("query_dividend_data", code=code, year=year, yearType=year_type))

    app.tool()(get_dividend_data)

    def get_adjust_factor_data(code: str, start_date: str, end_date: str) -> list[Record]:
        """Fetch adjustment factor data for calculating adjusted prices.

        Args:
            code: Stock code.
            start_date: 'YYYY-MM-DD'.
            end_date: 'YYYY-MM-DD'.

        """
        return df_to_records(bs.query("query_adjust_factor", code=code, start_date=start_date, end_date=end_date))

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
        start, end = lookback_range(_LATEST_TRADE_LOOKBACK_DAYS)
        df = bs.query(
            "query_history_k_data_plus",
            code=_LATEST_DATE_PROBE_CODE, fields="date",
            start_date=start, end_date=end,
            frequency="d", adjustflag="3",
        )
        return str(df["date"].max())

    app.tool()(get_latest_trading_date)

    def get_all_stock(date: str | None = None) -> list[Record]:
        """Fetch all stocks (A-shares + indices) and trading status.

        Args:
            date: Date 'YYYY-MM-DD'. Defaults to latest.

        """
        return df_to_records(bs.query("query_all_stock", day=date))

    app.tool()(get_all_stock)
