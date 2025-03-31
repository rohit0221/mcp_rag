import asyncio
import json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent

load_dotenv()  # Make sure your .env has OPENAI_API_KEY

async def test():
    # Load config from JSON file
    with open("../../../mcp_config/servers.json", "r") as f:
        config = json.load(f)

    print("\n🔌 Connecting to MCP Servers...\n")
    async with MultiServerMCPClient(config) as client:
        tools = client.get_tools()
        print(f"✅ Loaded {len(tools)} tools\n")

        agent = create_react_agent(ChatOpenAI(model="gpt-4o-mini"), tools)

        # 📁 Filesystem: List files
        print("📁 Filesystem: Listing files in the docs directory...\n")
        fs_result = await agent.ainvoke({"messages": "List files in the docs directory"})
        print("📄 File List Result:\n", fs_result)

        # 📄 Filesystem: Read file (use full path within allowed dir)
        print("\n📂 Filesystem: Reading file 'introduction.txt' (edit if needed)...\n")
        read_result = await agent.ainvoke({
            "messages": "Read file C:\\GitHub\\mcp_servers\\filesystem\\docs\\introduction.txt"
        })
        print("📑 File Read Result:\n", read_result)

        # 🧠 Chroma: List documents
        print("\n🔍 Chroma: Listing documents...\n")
        chroma_result = await agent.ainvoke({"messages": "List documents in Chroma"})
        print("🧠 Chroma Result:\n", chroma_result)

if __name__ == "__main__":
    asyncio.run(test())
