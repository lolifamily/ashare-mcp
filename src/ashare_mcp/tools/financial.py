"""Financial report tools: profitability, growth, balance sheet, cash flow, DuPont."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ashare_mcp.baostock_client import Record, df_to_records

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

    from ashare_mcp.baostock_client import Baostock


def register(app: FastMCP, bs: Baostock) -> None:
    """Register financial report tools with the MCP app."""

    def get_profit_data(code: str, year: str, quarter: int) -> list[Record]:
        """Fetch quarterly profitability data including netProfit, MBRevenue, totalShare.

        Args:
            code: Stock code, e.g. 'sh.600519'.
            year: 4-digit year, e.g. '2024'.
            quarter: 1, 2, 3, or 4.

        """
        return df_to_records(bs.query("query_profit_data", code=code, year=year, quarter=quarter))

    app.tool()(get_profit_data)

    def get_operation_data(code: str, year: str, quarter: int) -> list[Record]:
        """Fetch quarterly operation capability data (turnover ratios).

        Args:
            code: Stock code.
            year: 4-digit year.
            quarter: 1-4.

        """
        return df_to_records(bs.query("query_operation_data", code=code, year=year, quarter=quarter))

    app.tool()(get_operation_data)

    def get_growth_data(code: str, year: str, quarter: int) -> list[Record]:
        """Fetch quarterly growth data (YOY rates: YOYNI, YOYEPSBasic, etc).

        Args:
            code: Stock code.
            year: 4-digit year.
            quarter: 1-4.

        """
        return df_to_records(bs.query("query_growth_data", code=code, year=year, quarter=quarter))

    app.tool()(get_growth_data)

    def get_balance_data(code: str, year: str, quarter: int) -> list[Record]:
        """Fetch quarterly balance sheet ratios (currentRatio, liabilityToAsset, etc).

        Args:
            code: Stock code.
            year: 4-digit year.
            quarter: 1-4.

        """
        return df_to_records(bs.query("query_balance_data", code=code, year=year, quarter=quarter))

    app.tool()(get_balance_data)

    def get_cash_flow_data(code: str, year: str, quarter: int) -> list[Record]:
        """Fetch quarterly cash flow ratios (CFOToOR, CFOToNP, etc).

        Args:
            code: Stock code.
            year: 4-digit year.
            quarter: 1-4.

        """
        return df_to_records(bs.query("query_cash_flow_data", code=code, year=year, quarter=quarter))

    app.tool()(get_cash_flow_data)

    def get_dupont_data(code: str, year: str, quarter: int) -> list[Record]:
        """Fetch quarterly DuPont analysis data (ROE decomposition).

        Args:
            code: Stock code.
            year: 4-digit year.
            quarter: 1-4.

        """
        return df_to_records(bs.query("query_dupont_data", code=code, year=year, quarter=quarter))

    app.tool()(get_dupont_data)

    def get_performance_express_report(code: str, start_date: str, end_date: str) -> list[Record]:
        """Fetch performance express reports (absolute TotalAsset/NetAsset for some companies).

        Args:
            code: Stock code.
            start_date: 'YYYY-MM-DD'.
            end_date: 'YYYY-MM-DD'.

        """
        return df_to_records(bs.query(
            "query_performance_express_report",
            code=code, start_date=start_date, end_date=end_date,
        ))

    app.tool()(get_performance_express_report)

    def get_forecast_report(code: str, start_date: str, end_date: str) -> list[Record]:
        """Fetch performance forecast reports.

        Args:
            code: Stock code.
            start_date: 'YYYY-MM-DD'.
            end_date: 'YYYY-MM-DD'.

        """
        return df_to_records(bs.query(
            "query_forecast_report",
            code=code, start_date=start_date, end_date=end_date,
        ))

    app.tool()(get_forecast_report)
