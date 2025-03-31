# src/mcp_rag/chunking_and_embedding_agent/visualize_chunk_embed_graph.py

from rag_chat_graph import build_rag_chat_graph

def save_graph(graph, name: str):
    try:
        png_bytes = graph.get_graph().draw_mermaid_png()
        output_path = f"{name}.png"
        with open(output_path, "wb") as f:
            f.write(png_bytes)
        print(f"✅ Mermaid graph saved as: {output_path}")
    except Exception as e:
        print(f"❌ Could not render graph {name}: {e}")


def main():
    dummy_tools = {
        "read_file": None,
        "create_document": None,
        "search_similar": None
    }

    graphs = {
        "rag_chat_graph": build_rag_chat_graph(dummy_tools),
    }

    for name, graph in graphs.items():
        save_graph(graph, name)

if __name__ == "__main__":
    main()