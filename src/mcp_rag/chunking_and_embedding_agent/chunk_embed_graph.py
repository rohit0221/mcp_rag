# src/mcp_rag/chunking_and_embedding_agent/chunk_embed_graph.py

from typing import TypedDict, List, Dict
from langgraph.graph import StateGraph, END
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
import fitz  # PyMuPDF

class ChunkingState(TypedDict):
    file_paths: List[str]
    chunks: List[Dict]  # Dict[content, metadata]


async def parse_and_chunk_node(state: ChunkingState, tools) -> ChunkingState:
    read_tool = tools["read_file"]
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    embeddings = OpenAIEmbeddings()  # Requires OPENAI_API_KEY in .env

    all_chunks = []

    for path in state["file_paths"]:
        # 1. Read file using MCP
        raw_bytes = await read_tool.ainvoke({"path": path})
        if isinstance(raw_bytes, str):
            raw_bytes = raw_bytes.encode("latin1")  # Convert back to bytes

        # 2. Write to temp.pdf
        with open("temp.pdf", "wb") as f:
            f.write(raw_bytes)

        # 3. Extract text using PyMuPDF
        doc = fitz.open("temp.pdf")
        full_text = "\n".join([page.get_text() for page in doc])

        # 4. Chunk text
        chunks = text_splitter.split_text(full_text)

        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "content": chunk,
                "metadata": {
                    "source": path,
                    "chunk": i
                }
            })

    return {"file_paths": state["file_paths"], "chunks": all_chunks}


async def embed_and_store_node(state: ChunkingState, tools) -> ChunkingState:
    chroma_tool = tools["create_document"]
    embeddings = OpenAIEmbeddings()

    for chunk in state["chunks"]:
        vector = embeddings.embed_query(chunk["content"])
        await chroma_tool.ainvoke({
            "content": chunk["content"],
            "embedding": vector,
            "metadata": chunk["metadata"]
        })

    return state


def build_chunk_embed_graph(tools) -> StateGraph:
    graph = StateGraph(ChunkingState)

    graph.add_node("parse_and_chunk", lambda s: parse_and_chunk_node(s, tools))
    graph.add_node("embed_and_store", lambda s: embed_and_store_node(s, tools))

    graph.set_entry_point("parse_and_chunk")
    graph.add_edge("parse_and_chunk", "embed_and_store")
    graph.add_edge("embed_and_store", END)

    return graph.compile()
