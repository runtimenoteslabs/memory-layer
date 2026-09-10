"""Compatibility shim for the pre-3.0 ``MEMORY_LAYER_`` environment prefix.

Every setting moved to ``RUNTIME_MEMORY_`` when the package was renamed. Rather
than make each reader check two names, this copies any legacy variable onto its
new name once, at import, leaving an explicit setting untouched. Anything
already configured with the old prefix, such as an agent config file written
before the rename, keeps working.
"""

from __future__ import annotations

import os

LEGACY_PREFIX = "MEMORY_LAYER_"
PREFIX = "RUNTIME_MEMORY_"


def apply_legacy_env(environ: dict[str, str] | None = None) -> list[str]:
    """Copy ``MEMORY_LAYER_*`` variables onto their ``RUNTIME_MEMORY_*`` names.

    A variable already set under the new prefix wins, so an explicit setting is
    never overwritten by a stale one.

    Args:
        environ: Mapping to update. Defaults to ``os.environ``.

    Returns:
        The legacy names that were carried over, for callers that want to warn.
    """
    env = os.environ if environ is None else environ
    carried = []

    for name in [k for k in env if k.startswith(LEGACY_PREFIX)]:
        renamed = PREFIX + name[len(LEGACY_PREFIX) :]
        if renamed not in env:
            env[renamed] = env[name]
            carried.append(name)

    return carried
