"""runtime-memory as a Hermes Agent memory provider.

Hermes finds this package through the ``hermes_agent.memory_providers`` entry
point declared in ``pyproject.toml``, so installing runtime-memory into the Hermes
environment is enough to make it selectable:

    pip install git+https://github.com/runtimenoteslabs/memory-layer.git
    hermes config set memory.provider runtimememory

See ``docs/hermes.md`` for configuration and the trace format.
"""

from __future__ import annotations

from typing import Any

from runtime_memory.hermes._base import HERMES_AVAILABLE
from runtime_memory.hermes.provider import (
    PROVIDER_NAME,
    RuntimeMemoryProvider,
)


def register(ctx: Any) -> None:
    """Register the provider with Hermes' plugin loader.

    Args:
        ctx: The plugin context Hermes passes in. Only
            ``register_memory_provider`` is used.
    """
    ctx.register_memory_provider(RuntimeMemoryProvider())


__all__ = [
    "HERMES_AVAILABLE",
    "PROVIDER_NAME",
    "RuntimeMemoryProvider",
    "register",
]
