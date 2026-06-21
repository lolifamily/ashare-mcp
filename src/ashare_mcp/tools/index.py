"""Index constituent and industry classification tools."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ashare_mcp.utils import Record, df_to_records

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

    from ashare_mcp.baostock_client import Baostock


def register(app: FastMCP, bs: Baostock) -> None:
    """Register index-related tools with the MCP app."""

    def get_stock_industry(
        code: str,
        date: str | None = None,
    ) -> list[Record]:
        """Fetch industry classification data.

        Args:
            code: Stock code, e.g. 'sh.600519'.
            date: Optional date 'YYYY-MM-DD'. Defaults to latest.

        """
        return df_to_records(bs.query("query_stock_industry", code=code, date=date))

    app.tool()(get_stock_industry)

    def get_sz50_stocks(date: str | None = None) -> list[Record]:
        """Fetch SSE 50 (上证50) index constituent stocks.

        Args:
            date: Optional date 'YYYY-MM-DD'. Defaults to latest.

        """
        return df_to_records(bs.query("query_sz50_stocks", date=date))

    app.tool()(get_sz50_stocks)

    def get_hs300_stocks(date: str | None = None) -> list[Record]:
        """Fetch CSI 300 index constituent stocks.

        Args:
            date: Optional date 'YYYY-MM-DD'. Defaults to latest.

        """
        return df_to_records(bs.query("query_hs300_stocks", date=date))

    app.tool()(get_hs300_stocks)

    def get_zz500_stocks(date: str | None = None) -> list[Record]:
        """Fetch CSI 500 index constituent stocks.

        Args:
            date: Optional date 'YYYY-MM-DD'. Defaults to latest.

        """
        return df_to_records(bs.query("query_zz500_stocks", date=date))

    app.tool()(get_zz500_stocks)
