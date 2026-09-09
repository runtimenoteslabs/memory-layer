"""memory-layer as a Hermes Agent memory provider.

Hermes finds this package through the ``hermes_agent.memory_providers`` entry
point declared in ``pyproject.toml``, so installing memory-layer into the Hermes
environment is enough to make it selectable:

    pip install memory-layer
    hermes config set memory.provider memorylayer

See ``docs/hermes.md`` for configuration and the trace format.
"""

from __future__ import annotations

from typing import Any

from memory_layer.hermes._base import HERMES_AVAILABLE
from memory_layer.hermes.provider import (
    PROVIDER_NAME,
    MemoryLayerProvider,
)


def register(ctx: Any) -> None:
    """Register the provider with Hermes' plugin loader.

    Args:
        ctx: The plugin context Hermes passes in. Only
            ``register_memory_provider`` is used.
    """
    ctx.register_memory_provider(MemoryLayerProvider())


__all__ = [
    "HERMES_AVAILABLE",
    "PROVIDER_NAME",
    "MemoryLayerProvider",
    "register",
]
