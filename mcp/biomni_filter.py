"""MCP server for selecting relevant Biomni tools."""

from __future__ import annotations

import importlib.util
import json
import os
import re
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from dotenv import load_dotenv
from openai import OpenAI

try:
    from mcp.server import MCPServer
except ImportError:  # The public MCP SDK currently exposes FastMCP instead.
    from mcp.server.fastmcp import FastMCP as MCPServer


load_dotenv()


def _tools_path(uri: str) -> Path:
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


def _load_descriptions(uri: str) -> list[dict[str, str]]:
    source = _tools_path(uri)
    files = [source] if source.is_file() else sorted(source.glob("*.py"))
    descriptions: list[dict[str, str]] = []

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
            raise ValueError(f"{file} must define a list named 'description'")
        for entry in entries:
            if not isinstance(entry, dict):
                raise ValueError(f"{file} contains a non-object tool description")
            name = entry.get("name")
            description = entry.get("description")
            if not isinstance(name, str) or not isinstance(description, str):
                raise ValueError(f"{file} contains a tool without name or description")
            descriptions.append({"name": name, "description": description})
    return descriptions


def _json_content(content: str) -> Any:
    cleaned = content.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", cleaned, re.DOTALL)
    if fenced:
        cleaned = fenced.group(1)
    return json.loads(cleaned)


def _write_audit_record(
    request_id: str,
    prompt: str,
    model: str,
    tools_uri: str,
    selected_tools: list[dict[str, str | int]],
) -> Path:
    default_log_dir = Path(__file__).resolve().parent.parent / ".mcp-logs"
    log_dir = Path(os.environ.get("MCP_LOG_DIR", default_log_dir))
    log_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "request_id": request_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "prompt": prompt,
        "model": model,
        "tools_uri": tools_uri,
        "selected_tools": selected_tools,
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


def _selected_tools(
    prompt: str, descriptions: list[dict[str, str]], client: OpenAI, model: str
) -> list[dict[str, str | int]]:
    catalog = [
        {"index": index, "name": item["name"], "description": item["description"]}
        for index, item in enumerate(descriptions)
    ]
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": (
                    "Select the tools relevant to the user's request. Return only JSON "
                    'in the form {"indices":[0,1]}. Use an empty array when none apply. '
                    "Do not invent indices.\n\nTool catalog:\n"
                    + json.dumps(catalog, ensure_ascii=False)
                ),
            },
            {"role": "user", "content": prompt},
        ],
    )
    content = response.choices[0].message.content
    if not content:
        raise ValueError("OpenAI endpoint returned an empty tool selection")
    result = _json_content(content)
    indices = result.get("indices") if isinstance(result, dict) else None
    if not isinstance(indices, list) or not all(isinstance(index, int) for index in indices):
        raise ValueError("OpenAI endpoint returned invalid tool indices")

    valid_indices = set(indices)
    return [item for item in catalog if item["index"] in valid_indices]


server = MCPServer("biomni-tool-filter")


@server.tool()
def list_relevant_tools(prompt: str) -> str:
    """Return JSON metadata for tools relevant to a natural-language request."""
    request_id = uuid.uuid4().hex
    tools_uri = os.environ.get("TOOLS_URI")
    if not tools_uri:
        raise RuntimeError("TOOLS_URI is not configured")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured")
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    descriptions = _load_descriptions(tools_uri)
    client = OpenAI(
        api_key=api_key,
        base_url=os.environ.get("OPENAI_BASE_URL"),
    )
    selected_tools = _selected_tools(prompt, descriptions, client, model)
    log_path = _write_audit_record(
        request_id, prompt, model, tools_uri, selected_tools
    )
    print(
        f"MCP tool selection {request_id} logged to {log_path}",
        file=sys.stderr,
    )
    return json.dumps(selected_tools, ensure_ascii=False)


def main() -> None:
    print(
        f"Starting biomni-tool-filter; audit logs: "
        f"{os.environ.get('MCP_LOG_DIR', Path(__file__).resolve().parent.parent / '.mcp-logs')}",
        file=sys.stderr,
    )
    server.run(transport="stdio")


if __name__ == "__main__":
    main()