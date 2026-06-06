"""Macroeconomic data tools: interest rates, money supply."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ashare_mcp.baostock_client import Record, df_to_records

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

    from ashare_mcp.baostock_client import Baostock


def register(app: FastMCP, bs: Baostock) -> None:
    """Register macroeconomic data tools with the MCP app."""

    def get_deposit_rate_data(
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> list[Record]:
        """Fetch benchmark deposit rates within a date range.

        Args:
            start_date: Optional 'YYYY-MM-DD'.
            end_date: Optional 'YYYY-MM-DD'.

        """
        return df_to_records(bs.query("query_deposit_rate_data", start_date=start_date, end_date=end_date))

    app.tool()(get_deposit_rate_data)

    def get_loan_rate_data(
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> list[Record]:
        """Fetch benchmark loan rates within a date range.

        Args:
            start_date: Optional 'YYYY-MM-DD'.
            end_date: Optional 'YYYY-MM-DD'.

        """
        return df_to_records(bs.query("query_loan_rate_data", start_date=start_date, end_date=end_date))

    app.tool()(get_loan_rate_data)

    def get_required_reserve_ratio_data(
        start_date: str | None = None,
        end_date: str | None = None,
        year_type: str = "0",
    ) -> list[Record]:
        """Fetch required reserve ratio data.

        Args:
            start_date: Optional 'YYYY-MM-DD'.
            end_date: Optional 'YYYY-MM-DD'.
            year_type: '0' announcement date (default), '1' effective date.

        """
        return df_to_records(bs.query(
            "query_required_reserve_ratio_data",
            start_date=start_date, end_date=end_date, yearType=year_type,
        ))

    app.tool()(get_required_reserve_ratio_data)

    def get_money_supply_data_month(
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> list[Record]:
        """Fetch monthly money supply data (M0, M1, M2).

        Args:
            start_date: Optional 'YYYY-MM'.
            end_date: Optional 'YYYY-MM'.

        """
        return df_to_records(bs.query("query_money_supply_data_month", start_date=start_date, end_date=end_date))

    app.tool()(get_money_supply_data_month)

    def get_money_supply_data_year(
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> list[Record]:
        """Fetch yearly money supply data (M0, M1, M2 year-end balance).

        Args:
            start_date: Optional 'YYYY'.
            end_date: Optional 'YYYY'.

        """
        return df_to_records(bs.query("query_money_supply_data_year", start_date=start_date, end_date=end_date))

    app.tool()(get_money_supply_data_year)
