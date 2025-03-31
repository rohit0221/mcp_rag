from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
import tempfile
import base64
import re

class ChunkingState(TypedDict):
    file_paths: List[str]

async def chunk_and_embed_node(state: ChunkingState, tools) -> ChunkingState:
    read_file = tools["read_file"]
    create_document = tools["create_document"]
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    embeddings = OpenAIEmbeddings()

    for path in state["file_paths"]:
        print(f"📄 Reading file: {path}")

        # Read the base64-encoded string from the `.b64` file
        response = await read_file.ainvoke({"path": path})
        b64_string = response["text"] if isinstance(response, dict) else response

        # Decode the base64 string into raw PDF bytes
        raw_bytes = base64.b64decode(b64_string)

        # Write bytes to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(raw_bytes)
            temp_path = tmp_file.name

        # Load and split PDF
        loader = PyPDFLoader(temp_path)
        documents: List[Document] = loader.load_and_split(splitter)
        print(f"✂️ Chunked into {len(documents)} pieces.")

        # Embed and store
        # for doc in documents:
        #     vector = embeddings.embed_query(doc.page_content)
        #     await create_document.ainvoke({
        #         "content": doc.page_content,
        #         "embedding": vector,
        #         "metadata": doc.metadata
        #     })
        # Embed and store
        for i, doc in enumerate(documents):
            print(f"🔍 Processing chunk {i} of {len(documents)}")
            vector = embeddings.embed_query(doc.page_content)
            print(f"✅ Generated embedding for chunk {i}. Calling create_document...")
            await create_document.ainvoke({
                "content": doc.page_content,
                "embedding": vector,
                "metadata": doc.metadata
            })
            print(f"📝 Successfully stored chunk {i}.")


    print("✅ Done embedding and storing chunks.")
    return state

def build_chunk_embed_graph(tools):
    graph = StateGraph(ChunkingState)

    async def chunk_wrapper(state):
        return await chunk_and_embed_node(state, tools)

    graph.add_node("chunk_and_embed", chunk_wrapper)
    graph.set_entry_point("chunk_and_embed")
    graph.add_edge("chunk_and_embed", END)

    return graph.compile()
