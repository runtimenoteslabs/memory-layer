"""Hermes plugin base classes, with a standalone fallback.

The provider subclasses ``agent.memory_provider.MemoryProvider``, which only
exists inside a Hermes Agent installation. memory-layer is also installed on its
own, so importing this package must not require Hermes. When Hermes is absent we
fall back to a shim carrying the same surface, which keeps ``memory_layer.hermes``
importable and unit-testable anywhere.

The shim is deliberately a copy of the contract, not a reimplementation of it. If
Hermes changes the ABC, the real import is what the provider is validated
against; ``HERMES_AVAILABLE`` tells tests which one is in play.
"""

from __future__ import annotations

from typing import Any

try:
    from agent.memory_provider import (  # type: ignore[import-not-found]
        INDICATOR_GLYPH,
        MemoryProvider,
        RecallStatus,
        is_trivial_prompt,
    )

    HERMES_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only outside Hermes
    HERMES_AVAILABLE = False

    import re
    from abc import ABC, abstractmethod
    from dataclasses import dataclass

    INDICATOR_GLYPH = "\N{BRAIN}"

    @dataclass(frozen=True)
    class RecallStatus:  # type: ignore[no-redef]
        """What the last prefetch injected, for the recall indicator."""

        provider_label: str
        count: int
        glyph: str = INDICATOR_GLYPH

    _TRIVIAL_PROMPT_RE = re.compile(
        r"^(yes|no|ok|okay|sure|thanks|thank you|y|n|yep|nope|yeah|nah|"
        r"hi|hey|hello|yo|sup|"
        r"continue|go ahead|do it|proceed|got it|cool|nice|great|done|next|lgtm|k)"
        r"[\s!?.:;,\"'~]*$",
        re.IGNORECASE,
    )

    def is_trivial_prompt(text: str | None) -> bool:  # type: ignore[misc]
        """True for empty input, slash commands and bare acknowledgements."""
        stripped = (text or "").strip()
        if not stripped or stripped.startswith("/"):
            return True
        return bool(_TRIVIAL_PROMPT_RE.match(stripped))

    class MemoryProvider(ABC):  # type: ignore[no-redef]
        """Minimal stand-in for the Hermes memory provider contract.

        The optional hooks below are deliberately concrete no-ops rather than
        abstract methods: a provider overrides only the ones it needs, exactly as
        in the real contract this mirrors.
        """

        pre_compress_checkpoint_api_version = 1

        @property
        @abstractmethod
        def name(self) -> str: ...

        @abstractmethod
        def is_available(self) -> bool: ...

        @abstractmethod
        def initialize(self, session_id: str, **kwargs: Any) -> None: ...

        @abstractmethod
        def get_tool_schemas(self) -> list[dict[str, Any]]: ...

        def unavailable_reason(self) -> str:
            return ""

        def system_prompt_block(self) -> str:
            return ""

        def prefetch(self, query: str, *, session_id: str = "") -> str:
            return ""

        def queue_prefetch(self, query: str, *, session_id: str = "") -> None: ...

        def recall_status(self) -> RecallStatus | None:
            return None

        def sync_turn(
            self,
            user_content: str,
            assistant_content: str,
            *,
            session_id: str = "",
            messages: list[dict[str, Any]] | None = None,
        ) -> None: ...

        def handle_tool_call(self, tool_name: str, args: dict[str, Any], **kwargs: Any) -> str:
            raise NotImplementedError

        def shutdown(self) -> None: ...

        def on_turn_start(self, turn_number: int, message: str, **kwargs: Any) -> None: ...

        def on_session_end(self, messages: list[dict[str, Any]]) -> None: ...

        def on_session_switch(
            self,
            new_session_id: str,
            *,
            parent_session_id: str = "",
            reset: bool = False,
            rewound: bool = False,
            **kwargs: Any,
        ) -> None: ...

        def on_pre_compress(self, messages: list[dict[str, Any]]) -> str:
            return ""

        def on_delegation(
            self, task: str, result: str, *, child_session_id: str = "", **kwargs: Any
        ) -> None: ...

        def get_config_schema(self) -> list[dict[str, Any]]:
            return []

        def save_config(self, values: dict[str, Any], hermes_home: str) -> None: ...

        def on_memory_write(
            self,
            action: str,
            target: str,
            content: str,
            metadata: dict[str, Any] | None = None,
        ) -> None: ...

        def backup_paths(self) -> list[str]:
            return []


__all__ = [
    "HERMES_AVAILABLE",
    "INDICATOR_GLYPH",
    "MemoryProvider",
    "RecallStatus",
    "is_trivial_prompt",
]
