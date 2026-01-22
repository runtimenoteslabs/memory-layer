"""Unit tests for configuration file validation.

Tests for:
- hooks.json schema validation
- plugin.json schema validation
- .mcp.json schema validation
- Command file parsing
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

# Get the project root
PROJECT_ROOT = Path(__file__).parent.parent.parent


# =============================================================================
# JSON Schema Validation Helpers
# =============================================================================


def load_json_file(path: Path) -> dict[str, Any]:
    """Load and parse a JSON file."""
    with open(path) as f:
        return json.load(f)


def validate_hooks_json(data: dict[str, Any]) -> list[str]:
    """Validate hooks.json structure. Returns list of errors."""
    errors = []

    if "hooks" not in data:
        errors.append("Missing 'hooks' key")
        return errors

    hooks = data["hooks"]
    valid_events = {"SessionStart", "PreCompact", "SessionEnd", "PostToolUse"}

    for event_name, event_hooks in hooks.items():
        if event_name not in valid_events:
            errors.append(f"Invalid hook event: {event_name}")
            continue

        if not isinstance(event_hooks, list):
            errors.append(f"Hook event '{event_name}' must be a list")
            continue

        for i, hook_group in enumerate(event_hooks):
            if not isinstance(hook_group, dict):
                errors.append(f"Hook group {i} in '{event_name}' must be an object")
                continue

            if "hooks" not in hook_group:
                errors.append(f"Hook group {i} in '{event_name}' missing 'hooks' array")
                continue

            # PostToolUse should have a matcher
            if event_name == "PostToolUse" and "matcher" not in hook_group:
                errors.append(f"PostToolUse hook group {i} should have a 'matcher'")

            for j, hook in enumerate(hook_group["hooks"]):
                if "type" not in hook:
                    errors.append(f"Hook {j} in '{event_name}' missing 'type'")
                elif hook["type"] not in {"command", "script"}:
                    errors.append(f"Hook {j} in '{event_name}' has invalid type: {hook['type']}")

                if "command" not in hook:
                    errors.append(f"Hook {j} in '{event_name}' missing 'command'")

    return errors


def validate_plugin_json(data: dict[str, Any]) -> list[str]:
    """Validate plugin.json structure. Returns list of errors."""
    errors = []

    required_fields = ["name", "description", "version"]
    for field in required_fields:
        if field not in data:
            errors.append(f"Missing required field: {field}")

    if "version" in data:
        # Simple semver check
        version = data["version"]
        parts = version.split(".")
        if len(parts) < 2:
            errors.append(f"Invalid version format: {version}")

    if "capabilities" in data:
        caps = data["capabilities"]
        if not isinstance(caps, dict):
            errors.append("capabilities must be an object")
        else:
            valid_caps = {"hooks", "commands", "skills", "mcp"}
            for cap in caps:
                if cap not in valid_caps:
                    errors.append(f"Unknown capability: {cap}")

    if "engines" in data:
        engines = data["engines"]
        if not isinstance(engines, dict):
            errors.append("engines must be an object")

    return errors


def validate_mcp_json(data: dict[str, Any]) -> list[str]:
    """Validate .mcp.json structure. Returns list of errors."""
    errors = []

    if "mcpServers" not in data:
        errors.append("Missing 'mcpServers' key")
        return errors

    servers = data["mcpServers"]
    if not isinstance(servers, dict):
        errors.append("mcpServers must be an object")
        return errors

    for server_name, server_config in servers.items():
        if not isinstance(server_config, dict):
            errors.append(f"Server '{server_name}' config must be an object")
            continue

        if "command" not in server_config:
            errors.append(f"Server '{server_name}' missing 'command'")

        if "tools" in server_config:
            tools = server_config["tools"]
            if not isinstance(tools, list):
                errors.append(f"Server '{server_name}' tools must be an array")
            else:
                for i, tool in enumerate(tools):
                    if "name" not in tool:
                        errors.append(f"Tool {i} in '{server_name}' missing 'name'")
                    if "description" not in tool:
                        errors.append(f"Tool {i} in '{server_name}' missing 'description'")

    return errors


def parse_command_frontmatter(content: str) -> tuple[dict[str, Any], str]:
    """Parse YAML frontmatter from command markdown file.

    Returns:
        Tuple of (frontmatter dict, body content)
    """
    if not content.startswith("---"):
        return {}, content

    lines = content.split("\n")
    frontmatter_lines = []
    body_start = 0

    in_frontmatter = False
    for i, line in enumerate(lines):
        if line.strip() == "---":
            if not in_frontmatter:
                in_frontmatter = True
            else:
                body_start = i + 1
                break
        elif in_frontmatter:
            frontmatter_lines.append(line)

    # Simple YAML parsing (key: value only)
    frontmatter = {}
    for line in frontmatter_lines:
        if ":" in line:
            key, value = line.split(":", 1)
            frontmatter[key.strip()] = value.strip()

    body = "\n".join(lines[body_start:])
    return frontmatter, body


def validate_command_file(content: str) -> list[str]:
    """Validate command markdown file. Returns list of errors."""
    errors = []

    frontmatter, body = parse_command_frontmatter(content)

    if not frontmatter:
        errors.append("Missing YAML frontmatter")
    else:
        if "description" not in frontmatter:
            errors.append("Missing 'description' in frontmatter")

    if not body.strip():
        errors.append("Empty body content")

    # Check for required sections
    if "# " not in body:
        errors.append("Missing main heading")

    return errors


# =============================================================================
# hooks.json Tests
# =============================================================================


class TestHooksJsonValidation:
    """Tests for hooks.json schema validation."""

    @pytest.fixture
    def hooks_file_path(self) -> Path:
        """Path to hooks.json file."""
        return PROJECT_ROOT / "hooks" / "hooks.json"

    def test_hooks_file_exists(self, hooks_file_path: Path):
        """Test that hooks.json file exists (if plugin is installed)."""
        if not hooks_file_path.exists():
            pytest.skip("hooks.json not found - plugin not installed at project root")

    def test_hooks_file_valid_json(self, hooks_file_path: Path):
        """Test that hooks.json is valid JSON."""
        if not hooks_file_path.exists():
            pytest.skip("hooks.json not found")

        try:
            data = load_json_file(hooks_file_path)
            assert isinstance(data, dict)
        except json.JSONDecodeError as e:
            pytest.fail(f"Invalid JSON in hooks.json: {e}")

    def test_hooks_schema_validation(self, hooks_file_path: Path):
        """Test that hooks.json follows the expected schema."""
        if not hooks_file_path.exists():
            pytest.skip("hooks.json not found")

        data = load_json_file(hooks_file_path)
        errors = validate_hooks_json(data)

        assert not errors, f"Schema validation errors: {errors}"

    def test_hooks_has_session_start(self, hooks_file_path: Path):
        """Test that SessionStart hook is defined."""
        if not hooks_file_path.exists():
            pytest.skip("hooks.json not found")

        data = load_json_file(hooks_file_path)
        assert "SessionStart" in data.get("hooks", {})

    def test_hooks_has_pre_compact(self, hooks_file_path: Path):
        """Test that PreCompact hook is defined."""
        if not hooks_file_path.exists():
            pytest.skip("hooks.json not found")

        data = load_json_file(hooks_file_path)
        assert "PreCompact" in data.get("hooks", {})

    def test_hooks_has_session_end(self, hooks_file_path: Path):
        """Test that SessionEnd hook is defined."""
        if not hooks_file_path.exists():
            pytest.skip("hooks.json not found")

        data = load_json_file(hooks_file_path)
        assert "SessionEnd" in data.get("hooks", {})

    def test_hooks_has_post_tool_use(self, hooks_file_path: Path):
        """Test that PostToolUse hook is defined."""
        if not hooks_file_path.exists():
            pytest.skip("hooks.json not found")

        data = load_json_file(hooks_file_path)
        assert "PostToolUse" in data.get("hooks", {})

    def test_hooks_commands_reference_mem_cli(self, hooks_file_path: Path):
        """Test that hook commands reference the mem CLI."""
        if not hooks_file_path.exists():
            pytest.skip("hooks.json not found")

        data = load_json_file(hooks_file_path)
        hooks = data.get("hooks", {})

        for event_name, event_hooks in hooks.items():
            for hook_group in event_hooks:
                for hook in hook_group.get("hooks", []):
                    cmd = hook.get("command", "")
                    assert "mem " in cmd, f"Hook in {event_name} doesn't use mem CLI"

    def test_hooks_have_timeouts(self, hooks_file_path: Path):
        """Test that all hooks have timeout values."""
        if not hooks_file_path.exists():
            pytest.skip("hooks.json not found")

        data = load_json_file(hooks_file_path)
        hooks = data.get("hooks", {})

        for event_name, event_hooks in hooks.items():
            for hook_group in event_hooks:
                for hook in hook_group.get("hooks", []):
                    assert "timeout" in hook, f"Hook in {event_name} missing timeout"
                    assert isinstance(hook["timeout"], int)
                    assert hook["timeout"] > 0

    def test_invalid_hooks_detected(self):
        """Test that validation catches invalid hooks."""
        invalid_data = {
            "hooks": {
                "InvalidEvent": [{"hooks": [{"type": "command", "command": "echo test"}]}]
            }
        }
        errors = validate_hooks_json(invalid_data)
        assert any("Invalid hook event" in e for e in errors)

    def test_missing_command_detected(self):
        """Test that validation catches missing commands."""
        invalid_data = {
            "hooks": {
                "SessionStart": [{"hooks": [{"type": "command"}]}]
            }
        }
        errors = validate_hooks_json(invalid_data)
        assert any("missing 'command'" in e for e in errors)


# =============================================================================
# plugin.json Tests
# =============================================================================


class TestPluginJsonValidation:
    """Tests for plugin.json schema validation."""

    @pytest.fixture
    def plugin_file_path(self) -> Path:
        """Path to plugin.json file."""
        return PROJECT_ROOT / ".claude-plugin" / "plugin.json"

    def test_plugin_file_exists(self, plugin_file_path: Path):
        """Test that plugin.json file exists."""
        assert plugin_file_path.exists(), f"plugin.json not found at {plugin_file_path}"

    def test_plugin_file_valid_json(self, plugin_file_path: Path):
        """Test that plugin.json is valid JSON."""
        if not plugin_file_path.exists():
            pytest.skip("plugin.json not found")

        data = load_json_file(plugin_file_path)
        assert isinstance(data, dict)

    def test_plugin_schema_validation(self, plugin_file_path: Path):
        """Test that plugin.json follows the expected schema."""
        if not plugin_file_path.exists():
            pytest.skip("plugin.json not found")

        data = load_json_file(plugin_file_path)
        errors = validate_plugin_json(data)

        assert not errors, f"Schema validation errors: {errors}"

    def test_plugin_has_name(self, plugin_file_path: Path):
        """Test that plugin has a name."""
        if not plugin_file_path.exists():
            pytest.skip("plugin.json not found")

        data = load_json_file(plugin_file_path)
        assert "name" in data
        assert data["name"] == "memory-layer"

    def test_plugin_version_is_2x(self, plugin_file_path: Path):
        """Test that plugin version is 2.x."""
        if not plugin_file_path.exists():
            pytest.skip("plugin.json not found")

        data = load_json_file(plugin_file_path)
        assert "version" in data
        assert data["version"].startswith("2.")

    def test_plugin_has_all_capabilities(self, plugin_file_path: Path):
        """Test that plugin declares all required capabilities."""
        if not plugin_file_path.exists():
            pytest.skip("plugin.json not found")

        data = load_json_file(plugin_file_path)
        caps = data.get("capabilities", {})

        assert caps.get("hooks") is True
        assert caps.get("commands") is True
        assert caps.get("skills") is True
        assert caps.get("mcp") is True

    def test_plugin_requires_python(self, plugin_file_path: Path):
        """Test that plugin specifies Python dependency."""
        if not plugin_file_path.exists():
            pytest.skip("plugin.json not found")

        data = load_json_file(plugin_file_path)
        deps = data.get("dependencies", {})

        assert "python" in deps

    def test_invalid_plugin_detected(self):
        """Test that validation catches invalid plugin.json."""
        invalid_data = {"name": "test"}  # Missing required fields
        errors = validate_plugin_json(invalid_data)
        assert any("description" in e for e in errors)
        assert any("version" in e for e in errors)


# =============================================================================
# .mcp.json Tests
# =============================================================================


class TestMcpJsonValidation:
    """Tests for .mcp.json schema validation."""

    @pytest.fixture
    def mcp_file_path(self) -> Path:
        """Path to .mcp.json file."""
        return PROJECT_ROOT / ".mcp.json"

    def test_mcp_file_exists(self, mcp_file_path: Path):
        """Test that .mcp.json file exists."""
        assert mcp_file_path.exists(), f".mcp.json not found at {mcp_file_path}"

    def test_mcp_file_valid_json(self, mcp_file_path: Path):
        """Test that .mcp.json is valid JSON."""
        if not mcp_file_path.exists():
            pytest.skip(".mcp.json not found")

        data = load_json_file(mcp_file_path)
        assert isinstance(data, dict)

    def test_mcp_schema_validation(self, mcp_file_path: Path):
        """Test that .mcp.json follows the expected schema."""
        if not mcp_file_path.exists():
            pytest.skip(".mcp.json not found")

        data = load_json_file(mcp_file_path)
        errors = validate_mcp_json(data)

        assert not errors, f"Schema validation errors: {errors}"

    def test_mcp_has_memory_layer_server(self, mcp_file_path: Path):
        """Test that MCP config has memory-layer server."""
        if not mcp_file_path.exists():
            pytest.skip(".mcp.json not found")

        data = load_json_file(mcp_file_path)
        servers = data.get("mcpServers", {})

        assert "memory-layer" in servers

    def test_mcp_server_command(self, mcp_file_path: Path):
        """Test that MCP server uses correct command."""
        if not mcp_file_path.exists():
            pytest.skip(".mcp.json not found")

        data = load_json_file(mcp_file_path)
        server = data.get("mcpServers", {}).get("memory-layer", {})

        assert server.get("command") == "mem"
        assert "serve" in server.get("args", [])

    def test_mcp_has_required_tools(self, mcp_file_path: Path):
        """Test that MCP server declares required tools."""
        if not mcp_file_path.exists():
            pytest.skip(".mcp.json not found")

        data = load_json_file(mcp_file_path)
        server = data.get("mcpServers", {}).get("memory-layer", {})
        tools = server.get("tools", [])

        tool_names = {t["name"] for t in tools}
        required_tools = {"search_memories", "add_memory", "record_outcome", "get_context"}

        assert required_tools.issubset(tool_names), f"Missing tools: {required_tools - tool_names}"

    def test_invalid_mcp_detected(self):
        """Test that validation catches invalid .mcp.json."""
        invalid_data = {"mcpServers": {"test": {}}}  # Missing command
        errors = validate_mcp_json(invalid_data)
        assert any("missing 'command'" in e for e in errors)


# =============================================================================
# Command File Tests
# =============================================================================


class TestCommandFileValidation:
    """Tests for slash command file validation."""

    @pytest.fixture
    def commands_dir(self) -> Path:
        """Path to commands directory."""
        return PROJECT_ROOT / "commands"

    def test_commands_directory_exists(self, commands_dir: Path):
        """Test that commands directory exists (if plugin is installed)."""
        if not commands_dir.exists():
            pytest.skip("commands directory not found - plugin not installed at project root")

    def test_remember_command_exists(self, commands_dir: Path):
        """Test that remember.md exists."""
        if not commands_dir.exists():
            pytest.skip("commands directory not found")
        assert (commands_dir / "remember.md").exists()

    def test_recall_command_exists(self, commands_dir: Path):
        """Test that recall.md exists."""
        if not commands_dir.exists():
            pytest.skip("commands directory not found")
        assert (commands_dir / "recall.md").exists()

    def test_outcome_command_exists(self, commands_dir: Path):
        """Test that outcome.md exists."""
        if not commands_dir.exists():
            pytest.skip("commands directory not found")
        assert (commands_dir / "outcome.md").exists()

    def test_memory_context_command_exists(self, commands_dir: Path):
        """Test that memory-context.md exists."""
        if not commands_dir.exists():
            pytest.skip("commands directory not found")
        assert (commands_dir / "memory-context.md").exists()

    def test_command_has_frontmatter(self, commands_dir: Path):
        """Test that all command files have frontmatter."""
        if not commands_dir.exists():
            pytest.skip("commands directory not found")

        for cmd_file in commands_dir.glob("*.md"):
            content = cmd_file.read_text()
            frontmatter, _ = parse_command_frontmatter(content)

            assert frontmatter, f"{cmd_file.name} missing frontmatter"
            assert "description" in frontmatter, f"{cmd_file.name} missing description"

    def test_command_has_usage_section(self, commands_dir: Path):
        """Test that all command files have a usage section."""
        if not commands_dir.exists():
            pytest.skip("commands directory not found")

        for cmd_file in commands_dir.glob("*.md"):
            content = cmd_file.read_text().lower()
            assert "## usage" in content or "usage" in content, \
                f"{cmd_file.name} missing usage section"

    def test_command_validation(self, commands_dir: Path):
        """Test command file validation function."""
        if not commands_dir.exists():
            pytest.skip("commands directory not found")

        for cmd_file in commands_dir.glob("*.md"):
            content = cmd_file.read_text()
            errors = validate_command_file(content)

            assert not errors, f"{cmd_file.name} validation errors: {errors}"

    def test_frontmatter_parsing(self):
        """Test frontmatter parsing function."""
        content = """---
description: Test command
allowed-tools: Bash
---

# Test Command

Content here.
"""
        frontmatter, body = parse_command_frontmatter(content)

        assert frontmatter["description"] == "Test command"
        assert frontmatter["allowed-tools"] == "Bash"
        assert "# Test Command" in body

    def test_frontmatter_parsing_no_frontmatter(self):
        """Test parsing file without frontmatter."""
        content = "# Just a heading\n\nSome content."
        frontmatter, body = parse_command_frontmatter(content)

        assert frontmatter == {}
        assert "# Just a heading" in body

    def test_invalid_command_detected(self):
        """Test that validation catches invalid command files."""
        invalid_content = "Just plain text without any structure."
        errors = validate_command_file(invalid_content)

        assert any("frontmatter" in e for e in errors)


# =============================================================================
# Skills Directory Tests
# =============================================================================


class TestSkillsValidation:
    """Tests for skills directory structure."""

    @pytest.fixture
    def skills_dir(self) -> Path:
        """Path to skills directory."""
        return PROJECT_ROOT / "skills"

    def test_skills_directory_exists(self, skills_dir: Path):
        """Test that skills directory exists (if plugin is installed)."""
        if not skills_dir.exists():
            pytest.skip("skills directory not found - plugin not installed at project root")

    def test_memory_retrieval_skill_exists(self, skills_dir: Path):
        """Test that memory-retrieval skill exists."""
        if not skills_dir.exists():
            pytest.skip("skills directory not found")

        skill_dir = skills_dir / "memory-retrieval"
        assert skill_dir.exists(), "memory-retrieval skill not found"
        assert (skill_dir / "SKILL.md").exists(), "SKILL.md not found"

    def test_outcome_feedback_skill_exists(self, skills_dir: Path):
        """Test that outcome-feedback skill exists."""
        if not skills_dir.exists():
            pytest.skip("skills directory not found")

        skill_dir = skills_dir / "outcome-feedback"
        assert skill_dir.exists(), "outcome-feedback skill not found"
        assert (skill_dir / "SKILL.md").exists(), "SKILL.md not found"

    def test_coding_patterns_skill_exists(self, skills_dir: Path):
        """Test that coding-patterns skill exists."""
        if not skills_dir.exists():
            pytest.skip("skills directory not found")

        skill_dir = skills_dir / "coding-patterns"
        assert skill_dir.exists(), "coding-patterns skill not found"
        assert (skill_dir / "SKILL.md").exists(), "SKILL.md not found"

    def test_skill_files_have_content(self, skills_dir: Path):
        """Test that skill files have meaningful content."""
        if not skills_dir.exists():
            pytest.skip("skills directory not found")

        for skill_dir in skills_dir.iterdir():
            if skill_dir.is_dir():
                skill_file = skill_dir / "SKILL.md"
                if skill_file.exists():
                    content = skill_file.read_text()
                    assert len(content) > 100, f"{skill_file} has insufficient content"
                    assert "#" in content, f"{skill_file} missing headings"
