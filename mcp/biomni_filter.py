"""MCP server for selecting relevant Biomni tools."""

import asyncio
import importlib.util
import json
import sys
import uuid
from datetime import datetime, timezone
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
    mcp_log_dir: Path = Path(__file__).resolve().parent.parent / ".mcp-logs"


settings = Settings()  # type: ignore[call-arg]
mcp = MCPServer("biomni-tool-filter")


@mcp.tool()
async def list_relevant_tools(prompt: str) -> str:
    """Return JSON metadata for tools relevant to a natural-language request."""
    request_id = uuid.uuid4().hex
    descriptions = await asyncio.to_thread(_load_tools_descriptions, settings.tools_uri)
    async with AsyncOpenAI(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
    ) as client:
        selected_tools = await _select_relevant_items(
            prompt, descriptions, "tools", client, settings.openai_model
        )
    await _log_selection(
        request_id,
        prompt,
        settings.openai_model,
        settings.tools_uri,
        selected_tools,
        "selected_tools",
        "tools_uri",
    )
    return json.dumps(selected_tools, ensure_ascii=False)


@mcp.tool()
async def list_relevant_know_hows(prompt: str) -> str:
    """Return know-how documents relevant to a natural-language request."""
    request_id = uuid.uuid4().hex
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
    await _log_selection(
        request_id,
        prompt,
        settings.openai_model,
        settings.know_how_uri,
        selected_know_hows,
        "selected_know_hows",
        "know_how_uri",
    )
    return json.dumps(selected_know_hows, ensure_ascii=False)


def _load_know_hows_descriptions(uri: str) -> list[dict[str, str | int]]:
    source = _get_tools_path(uri)
    files = [source] if source.is_file() else sorted(source.glob("*.md"))
    know_hows: list[dict[str, str | int]] = []

    for file in files:
        if file.name.upper() in {"README.MD", "QUICK_START.MD"}:
            continue
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
        if file.name.startswith("_"):
            continue
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


# Remove later. Only for testing purposes.
async def _log_selection(
    request_id: str,
    prompt: str,
    model: str,
    source_uri: str,
    selected_items: list[dict[str, str | int]],
    selection_key: str = "selected_tools",
    source_key: str = "tools_uri",
) -> None:
    log_path = await asyncio.to_thread(
        _write_audit_record,
        request_id,
        prompt,
        model,
        source_uri,
        selected_items,
        selection_key,
        source_key,
    )
    print(
        f"MCP {selection_key.removeprefix('selected_')} selection "
        f"{request_id} logged to {log_path}",
        file=sys.stderr,
    )


# Remove later. Only for testing purposes.
def _write_audit_record(
    request_id: str,
    prompt: str,
    model: str,
    source_uri: str,
    selected_items: list[dict[str, str | int]],
    selection_key: str = "selected_tools",
    source_key: str = "tools_uri",
) -> Path:
    log_dir = settings.mcp_log_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "request_id": request_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "prompt": prompt,
        "model": model,
        source_key: source_uri,
        selection_key: selected_items,
    }
    output_path = log_dir / f"{request_id}.json"
    temporary_path = log_dir / f".{request_id}.tmp"
    try:
        temporary_path.write_text(
            json.dumps(record, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary_path.replace(output_path)
    except OSError as error:
        temporary_path.unlink(missing_ok=True)
        raise RuntimeError(f"Unable to write MCP audit log: {output_path}") from error
    return output_path


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
    print(
        f"Starting biomni-tool-filter; audit logs: {settings.mcp_log_dir}",
        file=sys.stderr,
    )
    mcp.run(transport="stdio")
