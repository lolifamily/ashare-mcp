# Conventions

Use `uv` for everything Python in this repo: `uv run`, `uv add`, `uv sync`, `uv run pytest`. Never call bare `python` or `pip` — it bypasses the locked env and desyncs `uv.lock`.

This applies to source/tests/deps only. Calling the server's `mcp__ashare__*` tools is unaffected — that's the product surface, not local tooling.

## Tool docstrings

Functions registered via `app.tool()(...)` ship their docstrings to MCP clients verbatim — that text is wire surface, not internal documentation. Keep them lean:

- ✅ arg semantics, return shape, null/error states, surprising data quirks the caller can't predict (e.g. "cumulative YTD, not single-quarter").
- ❌ design rationale, why-not-the-other-way, implementation choices, references to other approaches you considered, Linus-style asides.

The rejected items go in **code comments next to the lines they explain**, not in the docstring. A caller reading the tool description doesn't need to know why the threshold is 2 instead of 3; the next developer touching that line does.

This rule applies ONLY to `app.tool()`-registered functions. Module-level docstrings, class docstrings, and private helpers (`_foo`, inner pure functions) are developer-facing — write whatever helps the next reader.
