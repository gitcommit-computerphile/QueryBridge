from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    # full conversation history — add_messages merges rather than overwrites
    messages: Annotated[list, add_messages]
    # standalone rewritten question (history-aware)
    reformulated_query: str
    # router decision: "wiki" or "vectordb"
    route: str
    # retrieved context string
    context: str
