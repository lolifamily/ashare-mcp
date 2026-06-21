"""Index constituent and industry classification tools."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from ashare_mcp.utils import Record, df_to_records

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

    from ashare_mcp.baostock_client import Baostock, BsQueryFn

# index key -> baostock constituent query fn. The three constituent queries are
# structurally identical (same signature, same row shape), differing only in the
# query name, so they collapse into one tool selected by `index`. Keep the keys
# in sync with the Literal in get_index_constituents' signature (that Literal
# drives the tool's JSON schema; this table drives dispatch).
_INDEX_QUERIES: dict[str, BsQueryFn] = {
    "sz50": "query_sz50_stocks",
    "hs300": "query_hs300_stocks",
    "zz500": "query_zz500_stocks",
}


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

    def get_index_constituents(
        index: Literal["sz50", "hs300", "zz500"],
        date: str | None = None,
    ) -> list[Record]:
        """Fetch constituent stocks of a major A-share index.

        index:
          - 'sz50':  SSE 50 (上证50).
          - 'hs300': CSI 300 (沪深300).
          - 'zz500': CSI 500 (中证500).

        Args:
            index: Which index's constituents to fetch.
            date: Optional date 'YYYY-MM-DD'. Defaults to latest.

        """
        return df_to_records(bs.query(_INDEX_QUERIES[index], date=date))

    app.tool()(get_index_constituents)
