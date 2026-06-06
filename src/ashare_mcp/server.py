"""FastMCP server for A-Share market data."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import TYPE_CHECKING

from mcp.server.fastmcp import FastMCP

from ashare_mcp.baostock_client import Baostock
from ashare_mcp.tools import financial, index, macro, market, technical, valuation

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

type Lifespan = Callable[[FastMCP[None]], AbstractAsyncContextManager[None]]


def _register_all(app: FastMCP, bs: Baostock) -> None:
    market.register(app, bs)
    index.register(app, bs)
    macro.register(app, bs)
    financial.register(app, bs)
    valuation.register(app, bs)
    technical.register(app, bs)


def _make_lifespan(bs: Baostock) -> Lifespan:
    """Create a lifespan context manager that manages the baostock session."""
    @asynccontextmanager
    async def lifespan(_app: FastMCP[None]) -> AsyncGenerator[None]:
        bs.login()
        try:
            yield
        finally:
            bs.logout()

    return lifespan


def build_app(*, port: int | None = None) -> FastMCP:
    """Build an MCP app. Pass port for HTTP transport, omit for stdio."""
    bs = Baostock()
    lifespan = _make_lifespan(bs)
    if port is not None:
        app = FastMCP(name="ashare_mcp", lifespan=lifespan, port=port)
    else:
        app = FastMCP(name="ashare_mcp", lifespan=lifespan)
    _register_all(app=app, bs=bs)
    return app


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="A-Share MCP Server")
    parser.add_argument("--transport", choices=["stdio", "http"], default="stdio")
    parser.add_argument(
        "--port", type=int, default=3000,
        help="HTTP port (default 3000, ignored when --transport=stdio)",
    )
    args = parser.parse_args()

    if args.transport == "stdio":
        build_app().run(transport="stdio")
    else:
        build_app(port=args.port).run(transport="streamable-http")
