"""Sync-to-async bridge for the Hermes memory provider.

Every method on the Hermes ``MemoryProvider`` contract is synchronous, while
``MemoryEngine`` is fully async. The bridge owns one long-lived event loop on a
daemon thread and marshals coroutines onto it.

The loop is module-global and is deliberately never stopped by a provider's
``shutdown()``. Hermes builds one provider per agent and one agent per concurrent
chat session, so a per-provider teardown would strand every sibling provider's
engine on a dead loop. The loop is a daemon thread and dies with the process.
"""

from __future__ import annotations

import asyncio
import contextvars
import threading
from typing import TYPE_CHECKING, Any, TypeVar

from runtime_memory.core.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Coroutine

logger = get_logger(__name__)

T = TypeVar("T")

_loop: asyncio.AbstractEventLoop | None = None
_loop_thread: threading.Thread | None = None
_loop_lock = threading.Lock()

DEFAULT_TIMEOUT = 30.0
"""Seconds to wait for a bridged call before giving up."""


def get_loop() -> asyncio.AbstractEventLoop:
    """Return the shared background event loop, starting it if needed."""
    global _loop, _loop_thread

    with _loop_lock:
        if _loop is not None and _loop.is_running():
            return _loop

        loop = asyncio.new_event_loop()

        def _run() -> None:
            asyncio.set_event_loop(loop)
            loop.run_forever()

        thread = threading.Thread(target=_run, daemon=True, name="memory-layer-loop")
        thread.start()

        _loop, _loop_thread = loop, thread
        return loop


def run_sync(coro: Coroutine[Any, Any, T], timeout: float = DEFAULT_TIMEOUT) -> T:
    """Run *coro* on the shared loop and block until it finishes.

    Args:
        coro: The coroutine to run.
        timeout: Seconds to wait before raising ``TimeoutError``.

    Returns:
        Whatever the coroutine returned.

    Raises:
        TimeoutError: If the coroutine does not finish within *timeout*.
    """
    future = asyncio.run_coroutine_threadsafe(coro, get_loop())
    try:
        return future.result(timeout=timeout)
    except TimeoutError:
        future.cancel()
        raise


def spawn(coro: Coroutine[Any, Any, Any], *, label: str = "task") -> None:
    """Fire *coro* onto the shared loop without waiting for it.

    Used for the write path, which must not block the agent's reply. Failures are
    logged rather than raised, since there is no caller left to receive them.
    """
    future = asyncio.run_coroutine_threadsafe(coro, get_loop())

    def _report(fut: Any) -> None:
        try:
            fut.result()
        except Exception as exc:  # background work: log it, there is no caller left
            logger.warning(f"Background {label} failed: {exc}")

    future.add_done_callback(_report)


def context_thread(target: Any, name: str) -> threading.Thread:
    """Daemon thread running *target* inside a copy of the caller's context.

    Threads otherwise start with an empty ``contextvars`` context, which loses the
    profile scoping Hermes sets up for multi-profile installs.
    """
    return threading.Thread(
        target=contextvars.copy_context().run,
        args=(target,),
        daemon=True,
        name=name,
    )


def _reset_for_tests() -> None:
    """Stop and clear the shared loop. Test-only."""
    global _loop, _loop_thread

    with _loop_lock:
        if _loop is not None and _loop.is_running():
            _loop.call_soon_threadsafe(_loop.stop)
        if _loop_thread is not None:
            _loop_thread.join(timeout=5.0)
        _loop, _loop_thread = None, None
