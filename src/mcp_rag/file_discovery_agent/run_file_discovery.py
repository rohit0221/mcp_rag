import asyncio
import json
from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
from file_discovery_graph import build_file_discovery_graph

load_dotenv()

async def main():
    with open("../../../mcp_config/servers.json", "r") as f:
        config = json.load(f)

    async with MultiServerMCPClient(config) as client:
        all_tools = client.get_tools()
        for tool in all_tools:
            print(" -", tool.name)
        # tools = {k: v for k, v in all_tools.items() if k.startswith("filesystem.")}
        # tools = {tool.name: tool for tool in all_tools if tool.name.startswith("filesystem.")}
        tools = {tool.name: tool for tool in all_tools if tool.name in {"list_directory", "read_file"}}



        # tools = await client.get_tools(namespace="filesystem")
        graph = build_file_discovery_graph(tools)

        result = await graph.ainvoke({})
        print("📁 Discovered PDF Files:\n")
        for path in result["file_paths"]:
            print(f"🗎 {path}")


if __name__ == "__main__":
    asyncio.run(main())
