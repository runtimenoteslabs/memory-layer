"""Outcomes from a finished session, without the agent reporting them.

An outcome needs a verdict (did the session succeed?) and an attribution (did the
session act on this memory?). Agents rarely call an outcome tool: in the Tier 3
evaluation the agent never did, and never named a recalled memory's id unprompted.
This module reads both facts from the session's messages instead.

The verdict comes from the last test run in the tool results. The attribution comes
from ``Attributor`` implementations, each of which names the recalled memories the
session acted on and the evidence for each. A memory no attributor names gets no
outcome: applying a verdict to every recalled memory left the store worse than
recording nothing in three Tier 2 evaluation runs.

Messages are in the OpenAI format Hermes passes to memory providers: ``role`` and
``content``, with ``tool_calls`` on assistant messages and tool results as
``role: tool``.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

from runtime_memory.core.models import MemoryCategory, Outcome

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from runtime_memory.core.models import Memory

Message = dict[str, Any]
"""One OpenAI-format chat message."""

OWN_TOOL_PREFIX = "runtimememory_"
"""Calls to Runtime Memory's own tools. Their arguments name memories because the
tool asks for ids, which says nothing about whether the session acted on them."""


# =============================================================================
# Reading messages
# =============================================================================


def message_text(message: Message) -> str:
    """The text of a message's content, whether a string or a list of parts."""
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(part.get("text", "")) for part in content if isinstance(part, dict)
        )
    return ""


def tool_output(message: Message) -> str:
    """A tool result's output.

    Hermes wraps a terminal result as JSON with an ``output`` field; other tools
    return plain text. Both are read.
    """
    text = message_text(message)
    if text.startswith("{"):
        try:
            data = json.loads(text)
        except ValueError:
            return text
        if isinstance(data, dict):
            parts = [str(data[key]) for key in ("output", "stdout", "stderr") if data.get(key)]
            if parts:
                return "\n".join(parts)
    return text


def tool_calls(message: Message) -> list[tuple[str, str]]:
    """An assistant message's tool calls, as (function name, arguments text)."""
    calls = []
    for call in message.get("tool_calls") or []:
        function = call.get("function") or {}
        arguments = function.get("arguments", "")
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments)
        calls.append((str(function.get("name", "")), arguments))
    return calls


def transcript(messages: Sequence[Message]) -> str:
    """A session as text, one line per message, tool calls included.

    Before 4.0 the extraction transcript kept only message content, so the
    commands an agent ran, which live in ``tool_calls``, never reached it.

    Args:
        messages: The session's messages.

    Returns:
        ``role: text`` lines, with ``assistant called name: arguments`` for each
        tool call and tool results unwrapped from Hermes' JSON.
    """
    lines = []
    for message in messages:
        role = message.get("role", "?")
        text = tool_output(message) if role == "tool" else message_text(message)
        if text:
            lines.append(f"{role}: {text}")
        for name, arguments in tool_calls(message):
            lines.append(f"{role} called {name}: {arguments}")
    return "\n".join(lines)


# =============================================================================
# Verdict
# =============================================================================

_PYTEST = re.compile(r"\b(\d+) (passed|failed|errors?)\b")
_PYTEST_RUN = re.compile(r"\b\d+ (passed|failed|errors?)\b.*\bin \d+(\.\d+)?s\b")
_UNITTEST_RAN = re.compile(r"^Ran \d+ tests? in ", re.MULTILINE)
_UNITTEST_FAILED = re.compile(r"^FAILED \((failures|errors)=\d+", re.MULTILINE)
_UNITTEST_OK = re.compile(r"^OK\b", re.MULTILINE)
_JEST = re.compile(r"^Tests:\s+(.*\d+ (passed|failed).*)$", re.MULTILINE)
_CARGO = re.compile(r"^test result: (ok|FAILED)\.", re.MULTILINE)
_GO_FAIL = re.compile(r"^(--- FAIL:|FAIL\s)", re.MULTILINE)
_GO_OK = re.compile(r"^ok\s+\S+\s+\d", re.MULTILINE)


def run_verdict(output: str) -> Outcome | None:
    """The verdict of one tool result, if it is a test run's output.

    Recognises pytest, unittest, Jest, Cargo and Go. A run with any failure or
    error is ``FAILED``; a run with passes and no failures is ``WORKED``.

    Args:
        output: A tool result's output.

    Returns:
        The verdict, or None when the output is not a test run.
    """
    verdict: Outcome | None = None
    if _PYTEST_RUN.search(output):
        counts = {kind.rstrip("s"): int(n) for n, kind in _PYTEST.findall(output)}
        failed = counts.get("failed", 0) + counts.get("error", 0)
        verdict = Outcome.FAILED if failed else Outcome.WORKED
    elif _UNITTEST_RAN.search(output):
        if _UNITTEST_FAILED.search(output):
            verdict = Outcome.FAILED
        elif _UNITTEST_OK.search(output):
            verdict = Outcome.WORKED
    elif match := _JEST.search(output):
        verdict = Outcome.FAILED if "failed" in match.group(1) else Outcome.WORKED
    elif match := _CARGO.search(output):
        verdict = Outcome.WORKED if match.group(1) == "ok" else Outcome.FAILED
    elif _GO_FAIL.search(output):
        verdict = Outcome.FAILED
    elif _GO_OK.search(output):
        verdict = Outcome.WORKED
    return verdict


def session_verdict(messages: Sequence[Message]) -> Outcome | None:
    """The verdict of the last test run in a session's tool results.

    The last run is the one the session finished on, after any fixes made in
    response to earlier failures.

    Args:
        messages: The session's messages.

    Returns:
        The verdict, or None when the session ran no tests.
    """
    verdict = None
    for message in messages:
        if message.get("role") == "tool":
            verdict = run_verdict(tool_output(message)) or verdict
    return verdict


# =============================================================================
# Attribution
# =============================================================================


@dataclass(frozen=True)
class Attribution:
    """A recalled memory the session acted on, and why that is believed."""

    memory_id: str
    source: str
    """The attributor that found it, recorded as the outcome's trace origin."""

    evidence: str
    """A short quote from the session."""


class Attributor(Protocol):
    """Names the recalled memories a session acted on."""

    name: str

    def attribute(
        self, messages: Sequence[Message], recalled: Sequence[Memory]
    ) -> list[Attribution]:
        """Return an attribution for each recalled memory the session acted on."""
        ...


def _assistant_texts(messages: Sequence[Message]) -> Iterable[str]:
    """What the agent wrote and the arguments of the tools it called.

    Calls to Runtime Memory's own tools are left out: see ``OWN_TOOL_PREFIX``.
    """
    for message in messages:
        if message.get("role") != "assistant":
            continue
        yield message_text(message)
        for name, arguments in tool_calls(message):
            if not name.startswith(OWN_TOOL_PREFIX):
                yield arguments


def _quote(text: str, start: int, end: int, width: int = 60) -> str:
    """The text around a match, on one line."""
    snippet = text[max(0, start - width // 2) : end + width // 2]
    return " ".join(snippet.split())


class CitationAttributor:
    """A memory the agent names by id was acted on.

    The injected block carries each memory's id and asks the agent to name the
    ones it uses. An id counts when it appears in full, or as an unambiguous
    prefix of eight or more characters, in the agent's text or tool arguments.
    """

    name = "cited"

    def attribute(
        self, messages: Sequence[Message], recalled: Sequence[Memory]
    ) -> list[Attribution]:
        """Return an attribution for each recalled memory the agent cited."""
        ids = [memory.id for memory in recalled]
        found: dict[str, Attribution] = {}
        for text in _assistant_texts(messages):
            for match in re.finditer(r"\b[0-9a-f]{8}[0-9a-f-]*\b", text):
                token = match.group(0)
                candidates = [i for i in ids if i == token or i.startswith(token)]
                if len(candidates) == 1 and candidates[0] not in found:
                    found[candidates[0]] = Attribution(
                        candidates[0], self.name, _quote(text, match.start(), match.end())
                    )
        return list(found.values())


_QUOTED = re.compile(r"`([^`\n]{6,})`|'([^'\n]{6,})'|\"([^\"\n]{6,})\"")
# Command memories only. Troubleshooting and workaround memories quote the command
# that failed as well as the one that fixed it ("`python3 -m pytest` failed with
# No module named pytest"), so running a quoted command there can be the mistake the
# memory warns about. On the Tier 3 sessions every match in those two categories was
# of that kind.
_COMMAND_CATEGORIES = frozenset({MemoryCategory.COMMAND})


def _commands(memory: Memory) -> list[str]:
    """Quoted spans in a memory that look like commands: a space or a slash in them."""
    spans = (next(g for g in match.groups() if g) for match in _QUOTED.finditer(memory.content))
    return [" ".join(span.split()) for span in spans if " " in span or "/" in span]


class CommandAttributor:
    """A command memory whose command the agent ran was acted on.

    Applies to memories in the command category, and only to the quoted commands
    in them, compared with the agent's tool arguments once whitespace is
    normalised. It says nothing about other memories.
    """

    name = "command"

    def attribute(
        self, messages: Sequence[Message], recalled: Sequence[Memory]
    ) -> list[Attribution]:
        """Return an attribution for each recalled command memory the agent ran."""
        ran = [
            " ".join(arguments.split())
            for message in messages
            if message.get("role") == "assistant"
            for name, arguments in tool_calls(message)
            if not name.startswith(OWN_TOOL_PREFIX)
        ]
        found: list[Attribution] = []
        for memory in recalled:
            if memory.category not in _COMMAND_CATEGORIES:
                continue
            for command in _commands(memory):
                hit = next((call for call in ran if command in call), None)
                if hit is not None:
                    found.append(Attribution(memory.id, self.name, command))
                    break
        return found


DEFAULT_ATTRIBUTORS: tuple[Attributor, ...] = (CitationAttributor(), CommandAttributor())


def attribute(
    messages: Sequence[Message],
    recalled: Sequence[Memory],
    attributors: Sequence[Attributor] = DEFAULT_ATTRIBUTORS,
) -> list[Attribution]:
    """Combine attributors, keeping the first attribution found for each memory.

    Args:
        messages: The session's messages.
        recalled: The memories recalled in the session.
        attributors: Attributors, in order of preference.

    Returns:
        At most one attribution per memory.
    """
    found: dict[str, Attribution] = {}
    for attributor in attributors:
        for attribution in attributor.attribute(messages, recalled):
            found.setdefault(attribution.memory_id, attribution)
    return list(found.values())
