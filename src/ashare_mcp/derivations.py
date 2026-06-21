"""Derived financial metrics from baostock ratio fields."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ashare_mcp.utils import as_float, lookback_range

if TYPE_CHECKING:
    from ashare_mcp.baostock_client import Baostock
    from ashare_mcp.utils import Record

_PRICE_LOOKBACK_DAYS = 15
_OCF_PRECISION_PCT = 2.0


def get_latest_close(bs: Baostock, code: str) -> float:
    """Latest trading day close price (baostock has no real-time API)."""
    start, end = lookback_range(_PRICE_LOOKBACK_DAYS)
    df = bs.query(
        "query_history_k_data_plus",
        code=code,
        fields="date,close",
        start_date=start,
        end_date=end,
        frequency="d",
        adjustflag="3",
    )
    return as_float(df.loc[df["date"].idxmax(), "close"])


def derive_ocf(profit: Record, cash_flow: Record) -> tuple[float, float]:
    """OCF = MBRevenue * CFOToOR. Returns (value, precision_pct ~2%).

    Pure function: caller is responsible for fetching the two Records for the
    same (code, year, quarter).
    """
    return as_float(profit["MBRevenue"]) * as_float(cash_flow["CFOToOR"]), _OCF_PRECISION_PCT
