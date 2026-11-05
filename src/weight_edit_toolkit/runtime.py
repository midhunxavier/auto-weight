"""Configure and check an installed local MCP runtime."""
import asyncio
from importlib.util import find_spec
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

from .contracts import EditError


def config_path():
    config_home = Path(os.getenv("XDG_CONFIG_HOME") or Path.home() / ".config")
    return Path(os.getenv("AUTOWEIGHT_CONFIG") or config_home / "autoweight/editor.conf").expanduser().absolute()


async def server_tools(python, input_root, runs_dir):
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    parameters = StdioServerParameters(
        command=str(python),
        args=["-m", "weight_edit_toolkit.mcp_server",
              "--input-root", str(input_root), "--runs-dir", str(runs_dir)],
    )
    async with stdio_client(parameters) as (reader, writer):
        async with ClientSession(reader, writer) as client:
            async with asyncio.timeout(30):
                await client.initialize()
                tools = await client.list_tools()
    return sorted(tool.name for tool in tools.tools)


def setup_runtime(input_root, runs_dir):
    missing = [name for name in ("mcp", "torch", "transformers", "safetensors", "jsonschema")
               if find_spec(name) is None]
    if missing:
        raise EditError(f"missing runtime dependencies ({', '.join(missing)}). "
                        "Install the package with the [hf,mcp] extras.")

    input_root = input_root.expanduser().resolve(strict=True)
    if not input_root.is_dir():
        raise EditError("input root must be an existing directory")
    runs_dir = (runs_dir or Path.home() / ".local/share/autoweight/runs").expanduser().resolve()
    # Keep the virtual environment's interpreter path, even when it is a symlink.
    python = Path(sys.executable).absolute()
    paths = (python, input_root, runs_dir)
    if any("\n" in str(path) or "\r" in str(path) for path in paths):
        raise EditError("runtime, input, and output paths must fit on single config lines")

    try:
        tools = asyncio.run(server_tools(*paths))
    except (ExceptionGroup, ImportError, OSError, TimeoutError) as exc:
        raise EditError(f"could not start and query the local MCP server: {exc}") from exc
    if not tools:
        raise EditError("the local MCP server did not expose any tools")

    config = config_path()
    config.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".setup-", dir=config.parent) as staging:
        candidate = Path(staging) / "editor.conf"
        candidate.write_text("".join(f"{path}\n" for path in paths), encoding="utf-8")
        candidate.chmod(0o600)
        candidate.replace(config)
    return {"status": "ready", "config": str(config), "python": str(python),
            "input_root": str(input_root), "runs_dir": str(runs_dir), "tools": tools}
