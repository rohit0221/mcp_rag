import asyncio
import json
from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
from rag_chat_graph import build_rag_chat_graph

load_dotenv()

async def main():
    with open("../../../mcp_config/servers.json", "r") as f:
        config = json.load(f)

    async with MultiServerMCPClient(config) as client:
        all_tools = client.get_tools()
        tools = {
            tool.name: tool for tool in all_tools
            if tool.name in {"search_similar"}
        }

        print("\n🔌 Loaded MCP Tools:")
        for name in tools:
            print(f" - {name}")

        query = "What is the main contribution of the attention paper?"

        print(f"\n🤖 Asking: {query}")
        graph = build_rag_chat_graph(tools)

        result = await graph.ainvoke({"query": query})
        print("\n✅ Result:\n", result["answer"])


if __name__ == "__main__":
    asyncio.run(main())
