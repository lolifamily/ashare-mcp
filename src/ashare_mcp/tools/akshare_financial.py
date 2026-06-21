"""Financial statement tools backed by akshare (EastMoney)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from ashare_mcp.utils import Record, df_to_records

if TYPE_CHECKING:
    import pandas as pd
    from mcp.server.fastmcp import FastMCP

    from ashare_mcp.akshare_source import AkshareSource

_DEFAULT_PERIODS = 8


def _trim_statement(df: pd.DataFrame, periods: int) -> pd.DataFrame:
    """Keep the most-recent `periods` rows, then drop columns all-null within them.

    EastMoney returns the entire reporting history with hundreds of columns, a
    large fraction of which are all-null for any single stock — returning it
    whole is a huge, mostly-noise payload. Trimming rows first then dropping the
    now-empty columns is what shrinks the response. Caller is responsible for
    periods >= 1; DataFrame.head() reads negatives as "all rows except the last
    |n|", which would silently pull back the full history.
    """
    return df.head(periods).dropna(axis=1, how="all")


def register(app: FastMCP, src: AkshareSource) -> None:
    """Register akshare-backed financial statement tools."""

    def get_financial_statement(
        code: str,
        statement: Literal["balance", "income", "cash_flow"],
        periods: int = _DEFAULT_PERIODS,
    ) -> list[Record]:
        """Fetch a full financial statement (all periods, all columns) from EastMoney via akshare.

        `statement` selects which one (each is a separate EastMoney endpoint — one
        fetch per call, so request only the statement you need):
          - "balance"   资产负债表 — point-in-time snapshot per report date.
          - "income"    利润表 — key fields OPERATE_INCOME, NETPROFIT, ...
          - "cash_flow" 现金流量表 — key fields NETCASH_OPERATE (operating cash flow),
                        CONSTRUCT_LONG_ASSET (capex).

        Returns raw EastMoney field names (English uppercase keys), rows ordered
        most-recent-first (newest period at index 0). Columns entirely empty
        across the returned periods are dropped.

        IMPORTANT — for "income" and "cash_flow", quarterly values are cumulative
        YTD (year-to-date), not single-quarter:
          - Q1 (一季报) = 3 months
          - H1 (半年报) = 6 months
          - Q3 (三季报) = 9 months
          - FY (年报)   = 12 months
        Inspect REPORT_DATE_NAME before comparing across periods. To get a single
        quarter, subtract the previous cumulative period (e.g. Q3_single = Q3 - H1).
        The balance sheet is a snapshot, so this does not apply to it.

        Args:
            code: Stock code in baostock format, e.g. 'sh.600519'.
            statement: Which statement — "balance", "income", or "cash_flow".
            periods: How many most-recent report periods to return (default 8 ≈
                2 years of quarterly filings). Raise only when you need older
                periods — the full history is large.

        """
        fetch = {
            "balance": src.balance_sheet,
            "income": src.income_statement,
            "cash_flow": src.cash_flow,
        }
        # Literal isn't enforced at runtime — MCP clients can send any string.
        # Reject unknown values with the project's standard fail-loud ValueError
        # rather than letting the dict access raise a bare KeyError.
        if statement not in fetch:
            msg = f"statement must be one of {sorted(fetch)}; got {statement!r}"
            raise ValueError(msg)
        if periods < 1:
            msg = f"periods ({periods}) must be >= 1"
            raise ValueError(msg)
        return df_to_records(_trim_statement(fetch[statement](code), periods))

    app.tool()(get_financial_statement)

    def get_net_debt(code: str) -> dict[str, object]:
        """Compute net debt from the latest balance sheet (conservative upper bound).

        net_debt = SHORT_LOAN + LONG_LOAN + BOND_PAYABLE + NONCURRENT_LIAB_1YEAR - MONETARYFUNDS

        NONCURRENT_LIAB_1YEAR is a CN-GAAP aggregate bucket containing both
        long-term-debt current portion (interest-bearing) and lease-liability /
        long-payable current portions (not interest-bearing); the breakdown is
        not exposed by EastMoney. For lease-heavy or zero-leverage firms the
        returned net_debt is overstated by the non-interest portion of this
        bucket.

        Data source: EastMoney via akshare. Returns net_debt value, component
        breakdown, and the report date of the balance sheet used.

        Args:
            code: Stock code in baostock format, e.g. 'sh.600519'.

        """
        return src.net_debt(code)

    app.tool()(get_net_debt)
