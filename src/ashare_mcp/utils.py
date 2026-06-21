"""Shared pure utilities — no baostock/akshare dependency."""

from __future__ import annotations

import datetime as dt
import math
from typing import cast

import numpy as np
import pandas as pd
from dateutil.tz import gettz

_tz = gettz("Asia/Shanghai")
if _tz is None:
    _msg = "Asia/Shanghai timezone unavailable; is python-dateutil installed?"
    raise RuntimeError(_msg)
MARKET_TZ: dt.tzinfo = _tz

ZERO_THRESHOLD: float = 1e-9

type Record = dict[str, object]


def lookback_range(days: int, *, end: str | None = None) -> tuple[str, str]:
    """Return (start, end) 'YYYY-MM-DD' strings spanning `days` calendar days.

    end=None: anchor on today (Asia/Shanghai).
    end='YYYY-MM-DD': anchor on the user-supplied date.
    """
    if end is None:
        anchor = dt.datetime.now(tz=MARKET_TZ)
        end = anchor.strftime("%Y-%m-%d")
    else:
        anchor = dt.datetime.strptime(end, "%Y-%m-%d").replace(tzinfo=MARKET_TZ)
    start = (anchor - dt.timedelta(days=days)).strftime("%Y-%m-%d")
    return start, end


def scalar(v: object) -> object:
    """Normalize a pandas scalar for JSON: numpy scalar -> Python native, NaN -> None, date -> ISO string."""
    if isinstance(v, np.generic):  # np.int64/bool_ aren't json-native; .item() -> Python native
        v = cast("object", v.item())  # numpy stub types .item() as Any; pin it back to object
    if v is None:
        return None
    if isinstance(v, float):
        return None if math.isnan(v) else round(v, 10)
    if isinstance(v, dt.date):  # date/datetime/pd.Timestamp; pd.NaT is also a dt.date subclass
        return None if pd.isna(v) else v.isoformat()
    return v


def safe_float(val: object) -> float | None:
    """Coerce to float; None/NaN/str-unconvertible -> None.

    Normalizes through scalar() first: raw akshare DataFrame scalars reach here
    unnormalized, and np.int64 is NOT an int subclass, so a bare
    isinstance(val, int) check silently dropped integer columns to None.
    Routing through scalar() keeps the numpy / NaN rules in one place.
    """
    val = scalar(val)
    if isinstance(val, (int, float)):
        return float(val)  # scalar() already mapped NaN -> None, so this is finite
    if isinstance(val, str):
        try:
            return float(val)
        except ValueError:
            return None
    return None


def as_float(v: object) -> float:
    """Cast to float. Raises on None / NaN / unconvertible."""
    if v is None:
        msg = "value is None"
        raise TypeError(msg)
    if isinstance(v, float):
        if math.isnan(v):
            msg = "value is NaN"
            raise ValueError(msg)
        return v
    if isinstance(v, (int, str)):
        return float(v)
    msg = f"cannot convert to float: {type(v).__name__}"
    raise TypeError(msg)


def df_to_records(df: pd.DataFrame) -> list[Record]:
    """Convert a DataFrame to a list of dicts. NaN -> None, types preserved."""
    return [{str(k): scalar(v) for k, v in record.items()} for record in df.to_dict(orient="records")]
