import asyncio
import importlib.util
import json
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).parents[1] / "mcp" / "biomni_filter.py"
SPEC = importlib.util.spec_from_file_location("biomni_filter", MODULE_PATH)
assert SPEC is not None
assert SPEC.loader is not None
biomni_filter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(biomni_filter)


class FakeCompletions:
    def __init__(self, content: str) -> None:
        self.content = content
        self.request: dict[str, object] | None = None

    async def create(self, **kwargs: object) -> object:
        self.request = kwargs
        message = type("Message", (), {"content": self.content})()
        choice = type("Choice", (), {"message": message})()
        return type("Response", (), {"choices": [choice]})()


class FakeClient:
    def __init__(self, content: str) -> None:
        completions = FakeCompletions(content)
        self.completions = completions
        self.chat = type("Chat", (), {"completions": completions})()


def test_loads_tool_descriptions_from_file_uri(tmp_path: Path) -> None:
    source = tmp_path / "descriptions"
    source.mkdir()
    (source / "tools.py").write_text(
        "description = [{'name': 'tool_a', 'description': 'A tool'}]\n",
        encoding="utf-8",
    )
    (source / "_ignored.py").write_text("description = []\n", encoding="utf-8")

    assert biomni_filter._load_tools_descriptions(source.as_uri()) == [
        {"name": "tool_a", "description": "A tool"}
    ]


def test_loads_know_how_metadata_from_markdown(tmp_path: Path) -> None:
    source = tmp_path / "know_how"
    source.mkdir()
    (source / "README.md").write_text("# Ignore me\n", encoding="utf-8")
    (source / "guide.md").write_text(
        "# Example Guide\n\n## Metadata\n\n**Short Description**: A useful guide.\n",
        encoding="utf-8",
    )

    assert biomni_filter._load_know_hows_descriptions(source.as_uri()) == [
        {
            "index": 0,
            "title": "Example Guide",
            "short_description": "A useful guide.",
        }
    ]


def test_selects_items_by_model_indices() -> None:
    client = FakeClient('{"indices": [1]}')
    items = [
        {"name": "first", "description": "First item"},
        {"name": "second", "description": "Second item"},
    ]

    selected = asyncio.run(
        biomni_filter._select_relevant_items(
            "choose the second item", items, "tools", client, "test-model"
        )
    )

    assert selected == [{"index": 1, "name": "second", "description": "Second item"}]


def test_writes_selection_audit_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(biomni_filter.settings, "mcp_log_dir", tmp_path)

    output_path = biomni_filter._write_audit_record(
        "request-123",
        "find a guide",
        "test-model",
        "file:///know-how",
        [{"index": 0, "title": "Guide", "short_description": "Useful"}],
        "selected_know_hows",
        "know_how_uri",
    )

    assert output_path == tmp_path / "request-123.json"
    assert json.loads(output_path.read_text(encoding="utf-8")) == {
        "request_id": "request-123",
        "timestamp": json.loads(output_path.read_text(encoding="utf-8"))["timestamp"],
        "prompt": "find a guide",
        "model": "test-model",
        "know_how_uri": "file:///know-how",
        "selected_know_hows": [
            {"index": 0, "title": "Guide", "short_description": "Useful"}
        ],
    }
