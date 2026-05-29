from typing import Literal
from langchain_groq import ChatGroq
from langchain_community.retrievers import WikipediaRetriever
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from pydantic import BaseModel, Field

from graph.state import AgentState


# ---------------------------------------------------------------------------
# Shared LLM instance (reused across nodes)
# ---------------------------------------------------------------------------

def get_llm(temperature: float = 0.0) -> ChatGroq:
    return ChatGroq(model="llama-3.3-70b-versatile", temperature=temperature)


# ---------------------------------------------------------------------------
# Pydantic schema for structured router output
# ---------------------------------------------------------------------------

class RouteDecision(BaseModel):
    """Router decision for which data source to query."""
    datasource: Literal["wiki", "vectordb"] = Field(
        description=(
            "Choose 'wiki' for general knowledge, current events, people, places, "
            "or topics not likely in the internal knowledge base. "
            "Choose 'vectordb' for AI, ML, LLMs, LangChain, LangGraph, RAG, "
            "vector databases, Groq, or any technical AI/ML topic."
        )
    )
    reasoning: str = Field(description="One sentence explaining the routing decision.")


# ---------------------------------------------------------------------------
# Node 1: Reformulate query using chat history (history-aware pattern)
# ---------------------------------------------------------------------------

REFORMULATE_SYSTEM = """Given the conversation history and the latest user question, \
rewrite the question as a fully self-contained standalone question that can be \
understood without the conversation history. Do NOT answer it — only rewrite it. \
If the question is already standalone, return it unchanged."""


def reformulate_query(state: AgentState) -> dict:
    messages = state["messages"]
    llm = get_llm()

    if len(messages) <= 1:
        # first turn — no history to consider
        last_human = next(
            (m.content for m in reversed(messages) if isinstance(m, HumanMessage)), ""
        )
        return {"reformulated_query": last_human}

    # build history string excluding the last message
    history_lines = []
    for m in messages[:-1]:
        if isinstance(m, HumanMessage):
            history_lines.append(f"Human: {m.content}")
        elif isinstance(m, AIMessage):
            history_lines.append(f"Assistant: {m.content}")
    history_str = "\n".join(history_lines)

    last_question = messages[-1].content if messages else ""

    prompt = [
        SystemMessage(content=REFORMULATE_SYSTEM),
        HumanMessage(
            content=f"Conversation history:\n{history_str}\n\nLatest question: {last_question}"
        ),
    ]

    result = llm.invoke(prompt)
    return {"reformulated_query": result.content.strip()}


# ---------------------------------------------------------------------------
# Node 2: Route the reformulated query
# ---------------------------------------------------------------------------

ROUTER_SYSTEM = """You are an expert query router. \
Your job is to decide whether a user question should be answered using:
- 'vectordb': internal knowledge base covering AI, ML, LLMs, LangChain, LangGraph, \
RAG, vector databases, ChromaDB, Groq, transformers, prompt engineering, and AI agents.
- 'wiki': Wikipedia for general knowledge — history, science, geography, people, \
current events, or any topic NOT related to AI/ML tooling."""


def route_query(state: AgentState) -> dict:
    query = state["reformulated_query"]
    llm = get_llm()
    structured_llm = llm.with_structured_output(RouteDecision)

    prompt = [
        SystemMessage(content=ROUTER_SYSTEM),
        HumanMessage(content=f"Question: {query}"),
    ]

    decision: RouteDecision = structured_llm.invoke(prompt)
    print(f"[Router] '{query}' → {decision.datasource} | {decision.reasoning}")
    return {"route": decision.datasource}


# ---------------------------------------------------------------------------
# Node 3a: Wikipedia search
# ---------------------------------------------------------------------------

def wiki_search(state: AgentState) -> dict:
    query = state["reformulated_query"]
    try:
        retriever = WikipediaRetriever(top_k_results=3, doc_content_chars_max=2000)
        docs = retriever.invoke(query)
        context = "\n\n---\n\n".join(
            f"[Wikipedia: {d.metadata.get('title', 'Unknown')}]\n{d.page_content}"
            for d in docs
        )
        print(f"[WikiSearch] Retrieved {len(docs)} docs for: {query}")
        return {"context": context or "No Wikipedia results found."}
    except Exception as e:
        print(f"[WikiSearch] Failed: {e}")
        return {"context": "Wikipedia search failed. Answer from your own knowledge."}


# ---------------------------------------------------------------------------
# Node 3b: Vector DB search (injected at graph build time)
# ---------------------------------------------------------------------------

def make_vectordb_search(vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    def vectordb_search(state: AgentState) -> dict:
        query = state["reformulated_query"]
        docs = retriever.invoke(query)
        context = "\n\n---\n\n".join(
            f"[VectorDB: {d.metadata.get('title', 'Document')}]\n{d.page_content}"
            for d in docs
        )
        print(f"[VectorDB] Retrieved {len(docs)} chunks for: {query}")
        return {"context": context or "No results found in knowledge base."}

    return vectordb_search


# ---------------------------------------------------------------------------
# Node 4: Generate final answer
# ---------------------------------------------------------------------------

GENERATE_SYSTEM = """You are a helpful AI assistant. \
Answer the user's question using the retrieved context below. \
Be concise, accurate, and cite the source type (Wikipedia or VectorDB) when relevant. \
If the context does not contain enough information, say so honestly.

Retrieved Context:
{context}"""


def generate_answer(state: AgentState) -> dict:
    context = state.get("context", "")
    messages = state["messages"]
    llm = get_llm(temperature=0.3)

    system = SystemMessage(content=GENERATE_SYSTEM.format(context=context))
    response = llm.invoke([system] + list(messages))

    return {"messages": [AIMessage(content=response.content)]}


# ---------------------------------------------------------------------------
# Conditional edge: decide which retrieval branch to take
# ---------------------------------------------------------------------------

def decide_route(state: AgentState) -> Literal["wiki_search", "vectordb_search"]:
    return "wiki_search" if state["route"] == "wiki" else "vectordb_search"
