"""Tests for reading a session's verdict and attributions from its messages."""

from __future__ import annotations

import json

import pytest

from runtime_memory.core.attribution import (
    CitationAttributor,
    CommandAttributor,
    attribute,
    run_verdict,
    session_verdict,
    tool_output,
    transcript,
)
from runtime_memory.core.models import Memory, MemoryCategory, Outcome


def _assistant(text: str = "", calls: list[tuple[str, dict]] | None = None) -> dict:
    message = {"role": "assistant", "content": text}
    if calls:
        message["tool_calls"] = [
            {"id": f"call_{i}", "function": {"name": name, "arguments": json.dumps(args)}}
            for i, (name, args) in enumerate(calls)
        ]
    return message


def _tool(output: str, *, wrapped: bool = True) -> dict:
    content = json.dumps({"output": output, "exit_code": 0}) if wrapped else output
    return {"role": "tool", "content": content}


class TestRunVerdict:
    """Each test runner's summary line reads as a verdict."""

    @pytest.mark.parametrize(
        ("output", "verdict"),
        [
            ("..... \n5 passed in 0.12s", Outcome.WORKED),
            ("===== 1 failed, 4 passed in 0.30s =====", Outcome.FAILED),
            ("==== 2 passed, 1 error in 0.08s ====", Outcome.FAILED),
            ("Ran 3 tests in 0.001s\n\nOK", Outcome.WORKED),
            ("Ran 3 tests in 0.001s\n\nFAILED (failures=1)", Outcome.FAILED),
            ("Tests:       4 passed, 4 total", Outcome.WORKED),
            ("Tests:       1 failed, 3 passed, 4 total", Outcome.FAILED),
            ("test result: ok. 12 passed; 0 failed; 0 ignored", Outcome.WORKED),
            ("test result: FAILED. 11 passed; 1 failed", Outcome.FAILED),
            ("ok  \texample.com/ledger\t0.012s", Outcome.WORKED),
            ("--- FAIL: TestBalance (0.00s)\nFAIL\texample.com/ledger", Outcome.FAILED),
        ],
    )
    def test_recognised_runners(self, output: str, verdict: Outcome) -> None:
        assert run_verdict(output) == verdict

    @pytest.mark.parametrize(
        "output",
        ["total 12\ndrwxr-xr-x  2 me me 4096 .", "5 passed the review", "OK, done"],
    )
    def test_other_output_is_not_a_run(self, output: str) -> None:
        assert run_verdict(output) is None


class TestSessionVerdict:
    """The session finishes on its last test run."""

    def test_the_last_run_decides(self) -> None:
        messages = [_tool("1 failed, 4 passed in 0.3s"), _tool("5 passed in 0.2s")]

        assert session_verdict(messages) == Outcome.WORKED

    def test_output_after_the_last_run_does_not_clear_it(self) -> None:
        messages = [_tool("1 failed in 0.1s"), _tool("git status: clean")]

        assert session_verdict(messages) == Outcome.FAILED

    def test_no_tests_means_no_verdict(self) -> None:
        assert session_verdict([_assistant("Done."), _tool("ok")]) is None

    def test_plain_text_tool_results_are_read_too(self) -> None:
        assert session_verdict([_tool("3 passed in 0.1s", wrapped=False)]) == Outcome.WORKED

    def test_only_tool_results_count(self) -> None:
        """An agent saying the tests pass is not a test run."""
        assert session_verdict([_assistant("All 5 passed in 0.1s, done.")]) is None


def test_tool_output_unwraps_hermes_json() -> None:
    assert tool_output(_tool("5 passed in 0.1s")) == "5 passed in 0.1s"


def test_transcript_keeps_tool_calls() -> None:
    text = transcript(
        [
            {"role": "user", "content": "Run the tests"},
            _assistant(calls=[("terminal", {"command": ".venv/bin/python -m pytest -q"})]),
            _tool("5 passed in 0.1s"),
        ]
    )

    assert "user: Run the tests" in text
    assert 'assistant called terminal: {"command": ".venv/bin/python -m pytest -q"}' in text
    assert "tool: 5 passed in 0.1s" in text


def _memory(content: str, memory_id: str, category: MemoryCategory = MemoryCategory.PATTERN) -> Memory:
    return Memory(id=memory_id, content=content, category=category)


class TestCitations:
    """A memory named by id was acted on."""

    UV = _memory("Install with uv", "4c1d2e3f-0000-4000-8000-000000000001")
    PIP = _memory("Install with pip", "4c1d2e3f-0000-4000-8000-000000000002")
    CI = _memory("CI runs on push", "9a9f0000-0000-4000-8000-000000000003")

    def _found(self, *messages: dict) -> set[str]:
        attributions = CitationAttributor().attribute(list(messages), [self.UV, self.PIP, self.CI])
        return {a.memory_id for a in attributions}

    def test_a_full_id_in_the_reply(self) -> None:
        assert self._found(_assistant(f"Following `{self.UV.id}`, I used uv.")) == {self.UV.id}

    def test_a_unique_prefix(self) -> None:
        assert self._found(_assistant("Per 9a9f0000, CI will run it.")) == {self.CI.id}

    def test_an_ambiguous_prefix_names_nothing(self) -> None:
        """4c1d2e3f starts two ids."""
        assert self._found(_assistant("Per 4c1d2e3f, done.")) == set()

    def test_an_id_in_tool_arguments(self) -> None:
        call = _assistant(calls=[("write_file", {"content": f"# per {self.PIP.id}"})])
        assert self._found(call) == {self.PIP.id}

    def test_ids_passed_to_runtime_memorys_own_tools_do_not_count(self) -> None:
        call = _assistant(
            calls=[("runtimememory_outcome", {"outcome": "worked", "memory_ids": [self.UV.id]})]
        )
        assert self._found(call) == set()

    def test_an_id_in_a_tool_result_does_not_count(self) -> None:
        """Only what the agent wrote is a citation."""
        assert self._found(_tool(f"grep found {self.UV.id}")) == set()


class TestCommands:
    """A command memory whose command the agent ran was acted on."""

    RUN = _memory("Run the tests with `.venv/bin/python -m pytest -q`", "c0", MemoryCategory.COMMAND)
    WARNED = _memory(
        "`python3 -m pytest` fails with No module named pytest", "t0", MemoryCategory.TROUBLESHOOTING
    )

    def test_the_quoted_command_was_run(self) -> None:
        call = _assistant(calls=[("terminal", {"command": ".venv/bin/python  -m pytest -q tests/"})])

        (found,) = CommandAttributor().attribute([call], [self.RUN])

        assert found.memory_id == "c0"
        assert found.source == "command"

    def test_a_command_not_run_names_nothing(self) -> None:
        call = _assistant(calls=[("terminal", {"command": "pytest -q"})])
        assert CommandAttributor().attribute([call], [self.RUN]) == []

    def test_a_troubleshooting_memory_is_not_matched(self) -> None:
        """Its quoted command is the one that failed, and running it is the mistake."""
        call = _assistant(calls=[("terminal", {"command": "python3 -m pytest"})])
        assert CommandAttributor().attribute([call], [self.WARNED]) == []


def test_attribute_keeps_the_first_source_per_memory() -> None:
    memory = _memory("Run `.venv/bin/python -m pytest -q`", "c0ffee00-0000-4000-8000-00000000000a", MemoryCategory.COMMAND)
    messages = [
        _assistant(
            f"Using {memory.id}.",
            calls=[("terminal", {"command": ".venv/bin/python -m pytest -q"})],
        )
    ]

    (found,) = attribute(messages, [memory])

    assert found.source == "cited"
