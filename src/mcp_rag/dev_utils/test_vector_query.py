# src/mcp_rag/dev_utils/test_vector_query.py

import asyncio
import json
from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import OpenAIEmbeddings

load_dotenv()

async def main():
    # Load server config
    with open("../../../mcp_config/servers.json", "r") as f:
        config = json.load(f)

    # Setup client and get tools
    async with MultiServerMCPClient(config) as client:
        all_tools = client.get_tools()
        tools = {tool.name: tool for tool in all_tools if tool.name == "search_similar"}

        if "search_similar" not in tools:
            raise RuntimeError("search_similar tool not found. Is Chroma MCP running?")

        search_tool = tools["search_similar"]
        embeddings = OpenAIEmbeddings()

        # User query
        question = "What is the main contribution of the attention paper?"
        print(f"🔍 Querying: {question}")

        # Generate embedding
        embedding = embeddings.embed_query(question)

        # Call search_similar
        # Call search_similar
        raw_result = await search_tool.ainvoke({
            "embedding": embedding,
            "top_k": 3,
            "query": "What is Encoder and Decoder Stacks?"  # This is what was missing!
        })

        # Parse if needed
        if isinstance(raw_result, str):
            try:
                results = json.loads(raw_result)
            except Exception as e:
                print(f"❌ Failed to parse search_similar response: {e}")
                print("Raw response:")
                print(raw_result)
                return
        else:
            results = raw_result

        # Print results
        print("\n🧠 Top Matches:\n")
        for i, doc in enumerate(results.get("documents", [])):
            print(f"[{i+1}] {doc.get('metadata', {}).get('source', 'N/A')}")
            print(doc.get("content", "")[:500])
            print("---")

if __name__ == "__main__":
    asyncio.run(main())
