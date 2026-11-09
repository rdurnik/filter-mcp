"""MCP server for selecting relevant Biomni tools."""

import asyncio
import importlib.util
import json
from pathlib import Path
from urllib.parse import unquote, urlparse

from mcp.server import MCPServer
from openai import AsyncOpenAI
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    tools_uri: str = Field(min_length=1)
    know_how_uri: str = Field(min_length=1)
    openai_api_key: str = Field(min_length=1)
    openai_model: str = Field(min_length=1)
    openai_base_url: str | None = None


settings = Settings()  # type: ignore[call-arg]
mcp = MCPServer("biomni-tool-filter")


@mcp.tool()
async def list_relevant_tools(prompt: str) -> str:
    """Return JSON metadata for tools relevant to a natural-language request."""
    descriptions = await asyncio.to_thread(_load_tools_descriptions, settings.tools_uri)
    async with AsyncOpenAI(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
    ) as client:
        selected_tools = await _select_relevant_items(
            prompt, descriptions, "tools", client, settings.openai_model
        )

    return json.dumps(selected_tools, ensure_ascii=False)


@mcp.tool()
async def list_relevant_know_hows(prompt: str) -> str:
    """Return know-how documents relevant to a natural-language request."""
    know_hows = await asyncio.to_thread(
        _load_know_hows_descriptions, settings.know_how_uri
    )
    async with AsyncOpenAI(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
    ) as client:
        selected_know_hows = await _select_relevant_items(
            prompt, know_hows, "know-how documents", client, settings.openai_model
        )
    return json.dumps(selected_know_hows, ensure_ascii=False)


def _load_know_hows_descriptions(uri: str) -> list[dict[str, str | int]]:
    source = _get_tools_path(uri)
    files = [source] if source.is_file() else sorted(source.glob("*.md"))
    know_hows: list[dict[str, str | int]] = []

    for file in files:
        content = file.read_text(encoding="utf-8")
        title = next(
            line[2:].strip() for line in content.splitlines() if line.startswith("# ")
        )
        short_description = next(
            (
                line.split(":", 1)[1].strip()
                for line in content.splitlines()
                if line.lower().startswith("**short description**:")
            ),
        )
        know_hows.append(
            {
                "index": len(know_hows),
                "title": title,
                "short_description": short_description,
            }
        )
    return know_hows


def _load_tools_descriptions(uri: str) -> list[dict[str, str | int]]:
    source = _get_tools_path(uri)
    files = [source] if source.is_file() else sorted(source.glob("*.py"))
    descriptions: list[dict[str, str | int]] = []

    for file in files:
        module_name = f"_biomni_tool_descriptions_{file.stem}"
        spec = importlib.util.spec_from_file_location(module_name, file)
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load tool description file: {file}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        entries = getattr(module, "description", None)
        if not isinstance(entries, list):
            raise TypeError(f"{file} must define a list named 'description'")
        for entry in entries:
            if not isinstance(entry, dict):
                raise TypeError(f"{file} contains a non-object tool description")
            name = entry.get("name")
            description = entry.get("description")
            if not isinstance(name, str) or not isinstance(description, str):
                raise TypeError(f"{file} contains a tool without name or description")
            descriptions.append({"name": name, "description": description})
    return descriptions


async def _select_relevant_items(
    prompt: str,
    items: list[dict[str, str | int]],
    item_type: str,
    client: AsyncOpenAI,
    model: str,
) -> list[dict[str, str | int]]:
    catalog = [
        {
            "index": index,
            **{key: value for key, value in item.items() if key != "index"},
        }
        for index, item in enumerate(items)
    ]
    response = await client.chat.completions.create(
        model=model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": (
                    f"Select the {item_type} relevant to the user's request. "
                    'Return only JSON in the form {"indices":[0,1]}. '
                    "Use an empty array when none apply. Do not invent indices.\n\n"
                    f"{item_type.title()} catalog:\n"
                    + json.dumps(catalog, ensure_ascii=False)
                ),
            },
            {"role": "user", "content": prompt},
        ],
    )
    content = response.choices[0].message.content
    if not content:
        raise ValueError(f"OpenAI endpoint returned an empty {item_type} selection")
    result = json.loads(content)
    indices = result.get("indices") if isinstance(result, dict) else None
    if not isinstance(indices, list) or not all(
        isinstance(index, int) for index in indices
    ):
        raise ValueError(f"OpenAI endpoint returned invalid {item_type} indices")

    valid_indices = set(indices)
    return [item for item in catalog if item["index"] in valid_indices]


def _get_tools_path(uri: str) -> Path:
    parsed = urlparse(uri)
    if parsed.scheme != "file":
        raise ValueError(
            f"Unsupported tools URI scheme {parsed.scheme!r}; only file:// is supported"
        )
    if parsed.netloc not in ("", "localhost"):
        raise ValueError("Only local file URIs are supported")
    path = Path(unquote(parsed.path))
    if not path.exists():
        raise FileNotFoundError(f"Tools URI does not exist: {uri}")
    return path


if __name__ == "__main__":
    mcp.run(transport="stdio")
