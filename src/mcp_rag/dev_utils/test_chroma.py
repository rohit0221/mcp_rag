import asyncio
import json
from langchain_mcp_adapters.client import MultiServerMCPClient

chroma_only_config = {
    "chroma": {
        "command": "C:/Users/neuralninja7/.local/bin/uv.exe",
        "args": [
            "--directory",
            "C:/GitHub/mcp_servers/chroma",
            "run",
            "chroma"
        ],
        "transport": "stdio"
    }
}

async def test_chroma_tools():
    async with MultiServerMCPClient(chroma_only_config) as client:
        tools = client.get_tools()
        print(f"\n🔧 Found {len(tools)} tools from Chroma MCP:\n")
        for tool in tools:
            print(f"  - {tool.name}")

asyncio.run(test_chroma_tools())
