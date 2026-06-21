"""Exception hierarchy for data source operations."""


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
    """Raised when a baostock query returns zero usable rows for the caller."""

    def __init__(self, fn: str, kwargs: dict[str, object], reason: str | None = None) -> None:
        """Build a no-data error with the query context.

        `reason` overrides the default `no rows returned from {fn}()` message
        for cases where that phrasing would mislead -- e.g. the underlying
        query returned a populated table but the caller's filter (target
        absent from the result set, or empty industry field) yields no
        match. fn should still be the real baostock function that was
        called; reason explains why the result was unusable.
        """
        msg = reason or f"no rows returned from {fn}()"
        super().__init__(fn, kwargs, "EMPTY", msg)


class AkshareError(Exception):
    """Raised when an akshare call fails (network, anti-scrape, missing data, etc)."""

    def __init__(
        self,
        fn: str,
        kwargs: dict[str, object],
        cause: Exception,
        *,
        no_data: bool = False,
    ) -> None:
        """Store error context from an akshare call.

        no_data flags an empty-payload response (akshare returned nothing for this
        entity), as opposed to a transient fault. A caller that would otherwise fall
        back to a rough estimate uses it to refuse instead: an empty payload means
        the estimate's own inputs are equally unavailable.
        """
        self.fn = fn
        self.kwargs = kwargs
        self.cause = cause
        self.no_data = no_data
        super().__init__(f"akshare {fn}() failed: {cause!r} | params={kwargs}")
