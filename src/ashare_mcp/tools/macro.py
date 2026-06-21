"""Macroeconomic data tools: interest rates, reserve ratio, money supply."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from ashare_mcp.utils import Record, df_to_records

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

    from ashare_mcp.baostock_client import Baostock, BsQueryFn

# rate_type -> baostock query fn. deposit & loan rate queries share the same
# (start_date, end_date) signature and row-per-adjustment shape, so they collapse
# into one tool. required-reserve-ratio stays separate: it carries a year_type
# param the rate tables lack, and folding it in would add a parameter that is
# meaningless for deposit/loan. Keep keys in sync with the signature Literal.
_BENCHMARK_RATE_QUERIES: dict[str, BsQueryFn] = {
    "deposit": "query_deposit_rate_data",
    "loan": "query_loan_rate_data",
}

# freq -> baostock money-supply query fn. Same (start_date, end_date) signature;
# only the date granularity differs (see docstring). Keep keys in sync with the
# signature Literal.
_MONEY_SUPPLY_QUERIES: dict[str, BsQueryFn] = {
    "month": "query_money_supply_data_month",
    "year": "query_money_supply_data_year",
}


def register(app: FastMCP, bs: Baostock) -> None:
    """Register macroeconomic data tools with the MCP app."""

    def get_benchmark_rate_data(
        rate_type: Literal["deposit", "loan"],
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> list[Record]:
        """Fetch PBoC benchmark deposit or loan rates within a date range.

        rate_type:
          - 'deposit': benchmark deposit rates — demand plus fixed-term
                       3-month .. 5-year.
          - 'loan':    benchmark loan rates — 6-month .. above-5-year, plus
                       mortgage rates.

        Args:
            rate_type: Which rate table to fetch.
            start_date: Optional 'YYYY-MM-DD'.
            end_date: Optional 'YYYY-MM-DD'.

        """
        return df_to_records(bs.query(
            _BENCHMARK_RATE_QUERIES[rate_type], start_date=start_date, end_date=end_date,
        ))

    app.tool()(get_benchmark_rate_data)

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

    def get_money_supply_data(
        freq: Literal["month", "year"],
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> list[Record]:
        """Fetch money supply data (M0, M1, M2) at monthly or yearly frequency.

        freq:
          - 'month': monthly series — m0Month/m1Month/m2Month plus YOY and
                     ChainRelative. Dates are 'YYYY-MM'.
          - 'year':  year-end balances — m0Year/m1Year/m2Year plus YearYOY.
                     Dates are 'YYYY'.

        Args:
            freq: 'month' for the monthly series, 'year' for year-end balances.
            start_date: Optional. Format matches freq: 'YYYY-MM' (month) or 'YYYY' (year).
            end_date: Optional, same format as start_date.

        """
        return df_to_records(bs.query(
            _MONEY_SUPPLY_QUERIES[freq], start_date=start_date, end_date=end_date,
        ))

    app.tool()(get_money_supply_data)
