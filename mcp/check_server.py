"""Load the real MCP server and confirm its tools register.

test_mcp.py exercises the tool logic without the MCP SDK; this catches the other
failure — an SDK release that breaks `server.py` at import time. Exits nonzero on
any import error or if the tool set changes without this list being updated.
Run from this directory: python check_server.py
"""

import asyncio
import sys

EXPECTED = {
    "describe_checkpoint",
    "get_job",
    "get_job_status",
    "list_checkpoints",
    "list_my_jobs",
    "register_checkpoint",
    "submit_finetune",
}


async def _tools() -> set:
    import server

    return {t.name for t in await server.mcp.list_tools()}


def main() -> int:
    names = asyncio.run(_tools())
    if names != EXPECTED:
        print(f"tool set mismatch: missing={sorted(EXPECTED - names)} extra={sorted(names - EXPECTED)}")
        return 1
    print(f"ok  {len(names)} tools registered")
    return 0


if __name__ == "__main__":
    sys.exit(main())
