"""
Shared async-to-sync bridge for CrewAI tool threads.

CrewAI tools execute ``_run()`` synchronously, often inside worker threads
that *may or may not* already have a running event loop.  Calling
``asyncio.run()`` in such a thread raises
``RuntimeError: This event loop is already running``.

This module provides a single ``run_async()`` helper that every custom
tool should use instead of hand-rolling event-loop management.

Strategy:
    * Each thread gets its own ``asyncio`` event loop, created on first
      use and cached in ``threading.local()``.
    * ``run_until_complete()`` executes the coroutine on the thread-local
      loop.
    * The loop is never explicitly closed — it is reused across tool
      invocations within the same thread for the lifetime of the thread.

Usage::

    from agents.crews.tools.async_bridge import run_async

    class MyTool(BaseTool):
        def _run(self, query: str) -> str:
            result = run_async(some_async_function(query))
            return str(result)
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Awaitable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

_thread_local = threading.local()


def run_async(coro: Awaitable[T]) -> T:
    """Run an async coroutine from a synchronous CrewAI tool thread.

    Returns the coroutine's result.  Raises whatever the coroutine raises.

    The underlying event loop is created once per thread and reused,
    avoiding the overhead of ``asyncio.new_event_loop()`` on every call
    *and* the ``RuntimeError`` that ``asyncio.run()`` throws when a loop
    is already present.
    """
    loop: asyncio.AbstractEventLoop | None = getattr(_thread_local, "loop", None)

    if loop is None or loop.is_closed():
        loop = asyncio.new_event_loop()
        _thread_local.loop = loop
        logger.debug(
            "Created new event loop for thread %s",
            threading.current_thread().name,
        )

    return loop.run_until_complete(coro)
