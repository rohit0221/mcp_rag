import asyncio
import json
from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
from rag_chat_graph import build_rag_chat_graph

load_dotenv()

async def main():
    # Load MCP config
    with open("../../../mcp_config/servers.json", "r") as f:
        config = json.load(f)

    async with MultiServerMCPClient(config) as client:
        # Get only the tools needed for RAG
        all_tools = client.get_tools()
        tools = {
            tool.name: tool for tool in all_tools
            if tool.name in {"search_similar"}
        }

        print("\n🔌 Loaded MCP Tools:")
        for name in tools:
            print(f" - {name}")

        graph = build_rag_chat_graph(tools)

        print("\n💬 Ask your question below (type 'q' or 'quit' to exit)\n")

        while True:
            query = input("🤖 Your question: ").strip()
            if query.lower() in {"q", "quit"}:
                print("👋 Exiting. Bye!")
                break

            if not query:
                print("⚠️ Please enter a non-empty question.")
                continue

            print(f"\n🔍 Querying: {query}")
            try:
                result = await graph.ainvoke({"query": query})
                print("\n🧠 Answer:\n", result["answer"])
            except Exception as e:
                print(f"❌ Error while answering: {e}")


if __name__ == "__main__":
    asyncio.run(main())
