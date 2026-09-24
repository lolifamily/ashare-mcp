"""MCP server for A-Share market data."""

from __future__ import annotations

import argparse
import threading
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager, suppress
from typing import TYPE_CHECKING, Any

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError, UnexpectedToolError

from ashare_mcp.akshare_source import AKSHARE_AVAILABLE, AkshareSource
from ashare_mcp.baostock_client import Baostock
from ashare_mcp.errors import BaostockError
from ashare_mcp.tools import akshare_financial, financial, index, macro, market, technical, valuation

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from mcp.types import CallToolResult, InputRequiredResult

type Lifespan = Callable[[MCPServer[None]], AbstractAsyncContextManager[None]]


class AshareMCPServer(MCPServer[None]):
    """MCPServer with two defaults changed for this server's tools.

    1. Every registered tool defaults to unstructured output. This server's
       tools return baostock rows whose column set drifts (new fields appear,
       optional columns vanish), so the auto-generated outputSchema is mostly
       noise — it can't usefully validate a dict[str, object]. Meanwhile the
       structuredContent path duplicates the entire payload on the wire next
       to content[0].text, and every known consumer (Claude Desktop, Claude
       Code, LLM agents in general) reads content[0].text anyway. The
       structured copy is paid-for-and-ignored bytes. A tool can still opt
       back in by passing `structured_output=True` explicitly at registration.

    2. A tool exception's message reaches the caller. mcp 2.x sends only
       "Error executing tool <name>" for anything that isn't a ToolError, but
       our tools raise plain exceptions (BaostockError, AkshareError, the
       fail-loud ValueError) whose text is written for the caller: which query
       failed, with which params, or which argument is invalid. Without it an
       LLM caller can't correct its next call. mcp 1.x sent that text for every
       exception; this keeps that behavior.
    """

    def tool(self, *args: Any, **kw: Any) -> Callable[..., Any]:
        """Forward to MCPServer.tool with structured_output defaulted to False."""
        # *args/**kw passthrough so MCPServer can grow new tool() params (positional
        # or keyword) without us having to mirror the signature here. Only behavior
        # we override is the default for structured_output -- see class docstring
        # for the why.
        kw.setdefault("structured_output", False)
        return super().tool(*args, **kw)

    async def call_tool(self, *args: Any, **kw: Any) -> CallToolResult | InputRequiredResult:
        """Forward to MCPServer.call_tool, keeping the message of any tool exception."""
        try:
            return await super().call_tool(*args, **kw)
        except UnexpectedToolError as e:
            # e is "Error executing tool <name>"; __cause__ is what the tool raised.
            # Re-raised as a plain ToolError, the server returns this text as an
            # is_error result, formatted exactly as mcp 1.x did. It also logs it at
            # INFO rather than as a crash with a traceback, since nearly all of these
            # are anticipated failures (no data, bad params, source unreachable).
            msg = f"{e}: {e.__cause__}"
            raise ToolError(msg) from e.__cause__


def _register_all(app: MCPServer, bs: Baostock) -> None:
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


def _try_login(bs: Baostock) -> None:
    with suppress(BaostockError):
        bs.login()


def _make_lifespan(bs: Baostock) -> Lifespan:
    """Create a lifespan that warms up the baostock session and logs out on shutdown."""

    @asynccontextmanager
    async def lifespan(_app: MCPServer[None]) -> AsyncGenerator[None]:
        # Warm up off the handshake path; on failure the first query logs in itself.
        threading.Thread(target=_try_login, args=(bs,), daemon=True).start()
        try:
            yield
        finally:
            bs.logout()

    return lifespan


def build_app() -> MCPServer:
    """Build the MCP app. Transport settings (e.g. port) go to run(), not here."""
    bs = Baostock()
    app = AshareMCPServer(name="ashare_mcp", lifespan=_make_lifespan(bs))
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

    app = build_app()
    if args.transport == "stdio":
        app.run(transport="stdio")
    else:
        app.run(transport="streamable-http", port=args.port)
