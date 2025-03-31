import asyncio
import json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent

load_dotenv()

async def test():
    # Load MCP config
    with open("../../../mcp_config/servers.json", "r") as f:
        config = json.load(f)

    print("\n🔌 Connecting to MCP Servers...\n")
    async with MultiServerMCPClient(config) as client:
        tools = client.get_tools()
        print(f"✅ Loaded {len(tools)} tools in total.\n")

        # 🔧 Group tools by server name prefix
        grouped_tools = {}
        for tool in tools:
            # Extract server from tool.name: "filesystem.read_file" → "filesystem"
            if "." in tool.name:
                server_name = tool.name.split(".")[0]
            else:
                server_name = "unknown"
            grouped_tools.setdefault(server_name, []).append(tool)

        print("🔧 Tools grouped by MCP server:\n")
        for server, tool_list in grouped_tools.items():
            print(f"🔹 Server '{server}' - {len(tool_list)} tools:")
            for tool in tool_list:
                print(f"  - {tool.name}")
            print()

        # Create agent with all tools
        agent = create_react_agent(ChatOpenAI(model="gpt-4o-mini"), tools)

        # Test Filesystem tool
        print("📁 Filesystem: Listing files in the docs directory...\n")
        fs_result = await agent.ainvoke({"messages": "List files in the docs directory"})
        print("📄 File List Result:\n", fs_result)

        # Test reading a file
        print("\n📂 Filesystem: Reading file 'introduction.txt'...\n")
        read_result = await agent.ainvoke({
            "messages": "Read file C:\\GitHub\\mcp_servers\\filesystem\\docs\\introduction.txt"
        })
        print("📑 File Read Result:\n", read_result)

        # Test Chroma
        print("\n🔍 Chroma: Listing documents...\n")
        chroma_result = await agent.ainvoke({"messages": "List documents in Chroma"})
        print("🧠 Chroma Result:\n", chroma_result)


if __name__ == "__main__":
    asyncio.run(test())
