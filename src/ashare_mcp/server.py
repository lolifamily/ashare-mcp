"""FastMCP server for A-Share market data."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import TYPE_CHECKING, Any

from mcp.server.fastmcp import FastMCP

from ashare_mcp.akshare_source import AKSHARE_AVAILABLE, AkshareSource
from ashare_mcp.baostock_client import Baostock
from ashare_mcp.tools import akshare_financial, financial, index, macro, market, technical, valuation

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

type Lifespan = Callable[[FastMCP[None]], AbstractAsyncContextManager[None]]


class UnstructuredFastMCP(FastMCP):
    """FastMCP that defaults every registered tool to unstructured output.

    Why: this server's tools return baostock rows whose column set drifts (new
    fields appear, optional columns vanish), so the auto-generated outputSchema
    is mostly noise — it can't usefully validate a dict[str, object]. Meanwhile
    the structuredContent path duplicates the entire payload on the wire next
    to content[0].text, and every known consumer (Claude Desktop, Claude Code,
    LLM agents in general) reads content[0].text anyway. The structured copy
    is paid-for-and-ignored bytes.

    A tool can still opt back in by passing `structured_output=True` explicitly
    at registration; only the default flips.
    """

    def tool(self, *args: Any, **kw: Any) -> Callable[..., Any]:
        """Forward to FastMCP.tool with structured_output defaulted to False."""
        # *args/**kw passthrough so FastMCP can grow new tool() params (positional
        # or keyword) without us having to mirror the signature here. Only behavior
        # we override is the default for structured_output -- see class docstring
        # for the why.
        kw.setdefault("structured_output", False)
        return super().tool(*args, **kw)


def _register_all(app: FastMCP, bs: Baostock) -> None:
    market.register(app, bs)
    index.register(app, bs)
    macro.register(app, bs)
    financial.register(app, bs)
    technical.register(app, bs)
    if AKSHARE_AVAILABLE:
        src = AkshareSource()
        akshare_financial.register(app, src)
        valuation.register(app, bs, src)
    else:
        valuation.register(app, bs)


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
        app = UnstructuredFastMCP(name="ashare_mcp", lifespan=lifespan, port=port)
    else:
        app = UnstructuredFastMCP(name="ashare_mcp", lifespan=lifespan)
    _register_all(app=app, bs=bs)
    return app


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="A-Share MCP Server")
    parser.add_argument("--transport", choices=["stdio", "http"], default="stdio")
    parser.add_argument(
        "--port",
        type=int,
        default=3000,
        help="HTTP port (default 3000, ignored when --transport=stdio)",
    )
    args = parser.parse_args()

    if args.transport == "stdio":
        build_app().run(transport="stdio")
    else:
        build_app(port=args.port).run(transport="streamable-http")
