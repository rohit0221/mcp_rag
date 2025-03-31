from typing import TypedDict, List, Dict
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document


class RAGChatState(TypedDict):
    query: str
    documents: List[Document]
    answer: str


import json
from langchain_core.documents import Document

import re
from langchain_core.documents import Document

async def retrieve_docs_node(state: dict, tools) -> dict:
    query = state["query"]
    print(f"\n🔍 Querying: {query}")

    search_similar = tools["search_similar"]
    response = await search_similar.ainvoke({
        "query": query,
        "k": 5
    })

    if not isinstance(response, str):
        print("⚠️ Unexpected format from search_similar, expected string.")
        return state

    documents = []

    # Regex to find each result
    matches = re.findall(r"Content:\s*(.*?)\s+Metadata:\s*(\{.*?\})", response, re.DOTALL)
    for i, (content, metadata_str) in enumerate(matches):
        try:
            metadata = eval(metadata_str)  # 👈 Only if trusted input
            documents.append(Document(page_content=content.strip(), metadata=metadata))
        except Exception as e:
            print(f"❌ Failed to parse metadata for doc {i}: {e}")

    print(f"📚 Parsed {len(documents)} documents from response.")
    return {**state, "documents": documents}

async def generate_answer_node(state: RAGChatState, tools) -> RAGChatState:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    query = state["query"]
    docs = state["documents"]

    context = "\n\n".join([doc.page_content for doc in docs])
    prompt = f"""You are a helpful and knowledgeable assistant that only answers questions based strictly on the provided context.

                Your primary rule is:
                👉 If the context does NOT contain the answer, respond with:
                "I do not have enough information in the context to answer this question."

                NEVER use your own knowledge or make assumptions.

                ---

                📚 Context:
                {context}

                ❓ Question: {query}

💬 Answer:"""

    response = await llm.ainvoke(prompt)
    print("\n🧠 Final Answer:\n", response.content)
    return {**state, "answer": response.content}


def build_rag_chat_graph(tools):
    graph = StateGraph(RAGChatState)

    async def retrieve_wrapper(state):
        return await retrieve_docs_node(state, tools)

    async def answer_wrapper(state):
        return await generate_answer_node(state, tools)

    graph.add_node("retrieve", retrieve_wrapper)
    graph.add_node("generate_answer", answer_wrapper)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "generate_answer")
    graph.add_edge("generate_answer", END)

    return graph.compile()
