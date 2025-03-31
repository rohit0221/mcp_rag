# src/mcp_rag/chunking_and_embedding_agent/run_chunk_embed.py

import asyncio
import json
from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
from chunk_embed_graph import build_chunk_embed_graph

load_dotenv()

async def main():
    with open("../../../mcp_config/servers.json", "r") as f:
        config = json.load(f)

    async with MultiServerMCPClient(config) as client:
        all_tools = client.get_tools()
        tools = {
            tool.name: tool for tool in all_tools
            if tool.name in {"read_file", "create_document"}
        }

        print("\n🔌 Loaded MCP Tools:")
        for name in tools:
            print(f" - {name}")

        # Example input from file discovery agent, now pointing to the encoded PDF.
        file_paths = [
            "C:/GitHub/mcp_servers/filesystem/docs/pdfs_b64/attention.pdf.b64"
        ]

        print("\n📥 Processing files:")
        for path in file_paths:
            print(" ", path)

        graph = build_chunk_embed_graph(tools)
        await graph.ainvoke({"file_paths": file_paths})

if __name__ == "__main__":
    asyncio.run(main())
