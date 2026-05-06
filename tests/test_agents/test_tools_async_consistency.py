"""
Static analysis tests to enforce that ALL CrewAI tools use the shared
``run_async()`` bridge instead of hand-rolling event loop management.

These tests scan tool source files for forbidden patterns:
  - ``asyncio.run(`` — crashes when a loop is already running
  - ``asyncio.new_event_loop()`` — per-call overhead, inconsistent lifecycle
"""

from pathlib import Path

import pytest

TOOLS_DIR = Path("agents/crews/tools")

# Only scan actual tool files, not __init__.py or the bridge itself
TOOL_FILES = sorted(
    p
    for p in TOOLS_DIR.glob("*.py")
    if p.name not in ("__init__.py", "async_bridge.py")
)


@pytest.mark.parametrize("tool_path", TOOL_FILES, ids=lambda p: p.name)
def test_no_asyncio_run_in_tools(tool_path: Path):
    """No tool should use asyncio.run() — use run_async() instead."""
    source = tool_path.read_text(encoding="utf-8")
    assert "asyncio.run(" not in source, (
        f"{tool_path.name} uses asyncio.run() which crashes when an event loop "
        f"is already running.  Use run_async() from async_bridge instead."
    )


@pytest.mark.parametrize("tool_path", TOOL_FILES, ids=lambda p: p.name)
def test_no_manual_event_loop_in_tools(tool_path: Path):
    """No tool should create its own event loop — use run_async() instead."""
    source = tool_path.read_text(encoding="utf-8")
    assert "asyncio.new_event_loop()" not in source, (
        f"{tool_path.name} creates a manual event loop.  "
        f"Use run_async() from async_bridge instead."
    )


@pytest.mark.parametrize("tool_path", TOOL_FILES, ids=lambda p: p.name)
def test_tools_import_run_async_if_they_have_async_code(tool_path: Path):
    """If a tool contains 'await ' it should import run_async."""
    source = tool_path.read_text(encoding="utf-8")
    if "await " not in source:
        pytest.skip(f"{tool_path.name} has no async code")
    assert "from agents.crews.tools.async_bridge import run_async" in source, (
        f"{tool_path.name} has async code but doesn't import run_async"
    )
