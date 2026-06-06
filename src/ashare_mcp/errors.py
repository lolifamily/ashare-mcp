"""Exception hierarchy for baostock operations."""


class BaostockError(Exception):
    """Raised when a baostock API call returns an error code."""

    def __init__(self, fn: str, kwargs: dict[str, object], code: str, msg: str) -> None:
        """Store error context and build a descriptive message."""
        self.fn = fn
        self.kwargs = kwargs
        self.code = code
        self.msg = msg
        err = f"baostock {fn}() error {code}: {msg} | params={kwargs}"
        super().__init__(err)


class NoDataFoundError(BaostockError):
    """Raised when a baostock query returns zero rows."""

    def __init__(self, fn: str, kwargs: dict[str, object]) -> None:
        """Build a no-data error with the query context."""
        super().__init__(fn, kwargs, "EMPTY", f"no rows returned from {fn}()")
