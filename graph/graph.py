import sqlite3
from pathlib import Path
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver  # requires: pip install langgraph-checkpoint-sqlite

from graph.state import AgentState
from graph.nodes import (
    reformulate_query,
    route_query,
    wiki_search,
    make_vectordb_search,
    generate_answer,
    decide_route,
)
from vectordb.setup import get_vectorstore

SQLITE_DB_PATH = str(Path(__file__).parent.parent / "checkpoints.db")


def build_graph():
    vectorstore = get_vectorstore()
    vectordb_search = make_vectordb_search(vectorstore)

    builder = StateGraph(AgentState)

    # register nodes
    builder.add_node("reformulate_query", reformulate_query)
    builder.add_node("route_query", route_query)
    builder.add_node("wiki_search", wiki_search)
    builder.add_node("vectordb_search", vectordb_search)
    builder.add_node("generate_answer", generate_answer)

    # edges
    builder.set_entry_point("reformulate_query")
    builder.add_edge("reformulate_query", "route_query")
    builder.add_conditional_edges(
        "route_query",
        decide_route,
        {
            "wiki_search": "wiki_search",
            "vectordb_search": "vectordb_search",
        },
    )
    builder.add_edge("wiki_search", "generate_answer")
    builder.add_edge("vectordb_search", "generate_answer")
    builder.add_edge("generate_answer", END)

    # SqliteSaver gives us persistence across restarts
    conn = sqlite3.connect(SQLITE_DB_PATH, check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    graph = builder.compile(checkpointer=checkpointer)

    return graph, vectorstore
