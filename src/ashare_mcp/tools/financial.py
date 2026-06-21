"""Financial report tools: quarterly statements and performance/forecast reports."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from ashare_mcp.utils import Record, df_to_records

if TYPE_CHECKING:
    from collections.abc import Callable

    from mcp.server.fastmcp import FastMCP

    from ashare_mcp.baostock_client import Baostock, BsQueryFn


def _rederive_liability_to_asset(records: list[Record]) -> list[Record]:
    """In-place rewrite liabilityToAsset := 1 - 1/assetToEquity per row.

    baostock liabilityToAsset is off by 100x for reports published in a
    ~2024-08 to ~2026-04 window (6 quarters never backfilled). Same-class
    bug as modifyRecord.md 2019-11-23 entry on the same field; the 2024
    recurrence is undocumented. assetToEquity was never affected by either
    incident, so the inverse identity is reliable and matches the raw value
    to floating-point precision on clean rows, making unconditional
    rederivation safe — no time-window special case. Rows where
    assetToEquity is null/zero keep the raw liabilityToAsset value.
    """
    for r in records:
        a2e = r.get("assetToEquity")
        # exclude bool: True is an int subclass and would be treated as a2e=1.
        if isinstance(a2e, (int, float)) and not isinstance(a2e, bool) and a2e != 0:
            r["liabilityToAsset"] = 1 - 1 / a2e
    return records


def _identity(records: list[Record]) -> list[Record]:
    """Return records unchanged — default post-processor for reports needing no fix-up."""
    return records


# report -> (baostock query fn, post-processor). Keep keys in sync with the
# Literal in get_financial_indicators's signature (that Literal drives the tool's
# JSON schema; this table drives dispatch). The post-processor folds balance's
# liabilityToAsset fix-up into the data table, mirroring technical._INDICATORS,
# so the tool body stays branch-free — balance is no longer a special case.
_QUARTERLY_REPORTS: dict[str, tuple[BsQueryFn, Callable[[list[Record]], list[Record]]]] = {
    "profit": ("query_profit_data", _identity),
    "operation": ("query_operation_data", _identity),
    "growth": ("query_growth_data", _identity),
    "balance": ("query_balance_data", _rederive_liability_to_asset),
    "cash_flow": ("query_cash_flow_data", _identity),
    "dupont": ("query_dupont_data", _identity),
}

# kind -> baostock query fn for the two non-periodic performance disclosures.
# Same (code, start_date, end_date) signature; differ only in the row schema.
_PERFORMANCE_REPORTS: dict[str, BsQueryFn] = {
    "express": "query_performance_express_report",
    "forecast": "query_forecast_report",
}


def register(app: FastMCP, bs: Baostock) -> None:
    """Register financial report tools with the MCP app."""

    def get_financial_indicators(
        code: str,
        report: Literal["profit", "operation", "growth", "balance", "cash_flow", "dupont"],
        year: str,
        quarter: int,
    ) -> list[Record]:
        """Fetch a quarterly financial report; `report` selects which statement.

        baostock splits quarterly fundamentals across six statements, each with a
        distinct field set. All values are cumulative-from-year-start: quarter=1
        is 3-month, 2 is H1 (6-month), 3 is 9-month, 4 is FY (12-month).

        report:
          - 'profit':    profitability — roeAvg, npMargin, gpMargin, netProfit,
                         epsTTM, MBRevenue, totalShare.
          - 'operation': turnover ratios — NRTurnRatio, INVTurnRatio, CATurnRatio,
                         AssetTurnRatio (and matching *Days).
          - 'growth':    YoY growth rates — YOYNI, YOYEPSBasic, YOYEquity,
                         YOYAsset, YOYPNI.
          - 'balance':   balance-sheet ratios — currentRatio, quickRatio,
                         liabilityToAsset, assetToEquity. liabilityToAsset is
                         rederived as 1 - 1/assetToEquity to work around stale
                         upstream values (falls back to the raw value only when
                         assetToEquity is null or zero).
          - 'cash_flow': cash-flow ratios — CFOToOR, CFOToNP, CFOToGr, CAToAsset.
          - 'dupont':    DuPont ROE decomposition — dupontROE, dupontAssetTurn,
                         dupontNitogr, dupontTaxBurden, dupontEbittogr.

        Args:
            code: Stock code, e.g. 'sh.600519'.
            report: Which statement to fetch (see above).
            year: 4-digit year, e.g. '2024'.
            quarter: 1, 2, 3, or 4.

        """
        query_fn, postprocess = _QUARTERLY_REPORTS[report]
        return postprocess(df_to_records(bs.query(query_fn, code=code, year=year, quarter=quarter)))

    app.tool()(get_financial_indicators)

    def get_performance_report(
        code: str,
        kind: Literal["express", "forecast"],
        start_date: str,
        end_date: str,
    ) -> list[Record]:
        """Fetch performance express (业绩快报) or forecast (业绩预告) reports.

        kind:
          - 'express':  performance express reports — absolute
                        performanceExpressTotalAsset / NetAsset / EPSDiluted /
                        ROEWa / GRYOY for companies that file them ahead of the
                        full report.
          - 'forecast': performance forecast reports — profitForcastChgPctUp /
                        profitForcastChgPctDwn and the forecast type.

        Args:
            code: Stock code.
            kind: 'express' for filed express reports, 'forecast' for guidance.
            start_date: 'YYYY-MM-DD'.
            end_date: 'YYYY-MM-DD'.

        """
        return df_to_records(bs.query(
            _PERFORMANCE_REPORTS[kind],
            code=code, start_date=start_date, end_date=end_date,
        ))

    app.tool()(get_performance_report)
