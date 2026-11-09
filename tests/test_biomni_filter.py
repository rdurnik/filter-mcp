import asyncio
import importlib.util
from pathlib import Path

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
    (source / "_extra.py").write_text(
        "description = [{'name': 'tool_b', 'description': 'Another tool'}]\n",
        encoding="utf-8",
    )

    assert biomni_filter._load_tools_descriptions(source.as_uri()) == [
        {"name": "tool_b", "description": "Another tool"},
        {"name": "tool_a", "description": "A tool"},
    ]


def test_loads_know_how_metadata_from_markdown(tmp_path: Path) -> None:
    source = tmp_path / "know_how"
    source.mkdir()
    (source / "README.md").write_text(
        "# README\n\n**Short Description**: Project documentation.\n",
        encoding="utf-8",
    )
    (source / "guide.md").write_text(
        "# Example Guide\n\n## Metadata\n\n**Short Description**: A useful guide.\n",
        encoding="utf-8",
    )

    assert biomni_filter._load_know_hows_descriptions(source.as_uri()) == [
        {
            "index": 0,
            "title": "README",
            "short_description": "Project documentation.",
        },
        {
            "index": 1,
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
