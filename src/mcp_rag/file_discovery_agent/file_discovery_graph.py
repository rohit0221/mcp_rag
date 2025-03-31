from typing import TypedDict, List
from langgraph.graph import StateGraph, END

class FileDiscoveryState(TypedDict):
    file_paths: List[str]

async def list_pdfs_node(state: FileDiscoveryState, tools) -> FileDiscoveryState:
    list_tool = tools["list_directory"]
    base_path = "C:\\GitHub\\mcp_servers\\filesystem\\docs"
    result = await list_tool.ainvoke({"path": base_path})

    # Parse raw text lines like: "[FILE] name.pdf"
    if isinstance(result, str):
        lines = result.splitlines()
        pdf_files = [line.replace("[FILE]", "").strip() for line in lines if line.strip().endswith(".pdf")]
        full_paths = [f"{base_path}\\{name}" for name in pdf_files]
        return {"file_paths": full_paths}

    raise ValueError("Unexpected response from list_directory")

def build_file_discovery_graph(tools):
    graph = StateGraph(FileDiscoveryState)

    async def list_wrapper(state):
        return await list_pdfs_node(state, tools)

    graph.add_node("list_pdfs", list_wrapper)

    graph.set_entry_point("list_pdfs")
    graph.add_edge("list_pdfs", END)

    return graph.compile()
