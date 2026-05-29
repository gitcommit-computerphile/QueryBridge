# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the App

```bash
# activate virtual env first
.venv\Scripts\activate

# run the app
python app.py
```

Opens at `http://localhost:7860`. On first run, ChromaDB initialises and downloads the `all-MiniLM-L6-v2` embedding model (~90MB, one time only).

## Environment

Requires `GROQ_API_KEY` in `.env`:
```
GROQ_API_KEY=gsk_...
```

## Architecture

This is a LangGraph RAG application with an LLM-based router that decides whether to answer from Wikipedia or a local vector database.

### Request flow

```
User message (app.py)
  → reformulate_query  — rewrites to standalone question using conversation history
  → route_query        — Groq structured output picks "wiki" or "vectordb"
  → wiki_search        — WikipediaRetriever (top 3 articles, max 2000 chars each)
    OR vectordb_search — ChromaDB similarity search (top 4 chunks)
  → generate_answer    — Groq LLM with context injected into system prompt + full message history
```

### Key design decisions

**Router uses Pydantic structured output** (`RouteDecision` in `graph/nodes.py`): forces the LLM to return exactly `"wiki"` or `"vectordb"` — no free-text parsing. The router prompt defines VectorDB as covering AI/ML topics (LangChain, LangGraph, RAG, Groq, etc.) and Wiki for everything else.

**Reformulation happens before routing**: queries with pronouns or references to prior turns (e.g. "what about their training data?") are rewritten as fully self-contained questions first. This ensures both the router and retriever receive a clean, unambiguous query. First-turn queries skip the LLM call and pass through unchanged.

**Context injected into system prompt, not as a message**: `generate_answer` formats retrieved docs into the system prompt via `GENERATE_SYSTEM.format(context=context)`, then appends the full `messages` history. The LLM sees context as background knowledge, not user input.

**`make_vectordb_search` is a factory** (`graph/nodes.py`): the vectorstore object is created in `graph/graph.py` and passed in via closure to avoid circular imports between `nodes.py` and `vectordb/setup.py`.

**Persistence**: `SqliteSaver` with a raw `sqlite3.connect(..., check_same_thread=False)` connection writes checkpoints to `checkpoints.db`. Each Gradio session has a UUID `thread_id` stored in `gr.State(value=lambda: str(uuid.uuid4()))` — the lambda ensures each browser tab gets a distinct ID. Delete `checkpoints.db` to clear all history.

**Multi-session support**: `app.py` exposes three session controls in the sidebar: the current session ID (read-only), a Radio list of all past `thread_id`s queried directly from `checkpoints.db`, and a "New Session" button. Selecting a past session calls `graph.get_state(config)` to reconstruct the full message history and re-render it in the Gradio chatbot. The session list refreshes automatically after each message and can be refreshed manually.

**`add_messages` reducer** on `AgentState.messages` appends rather than overwrites — each node only needs to return new messages; LangGraph merges them automatically.

### Storage locations

| Path | Contents |
|---|---|
| `chroma_db/` | Persistent ChromaDB vector store |
| `checkpoints.db` | LangGraph SQLite conversation checkpoints |
| `data/sample_docs.txt` | Seed documents loaded into ChromaDB on first run (AI/ML topics) |

### LLM

All nodes use `ChatGroq(model="llama-3.3-70b-versatile")`. Router and reformulation use `temperature=0.0`; generation uses `temperature=0.3`. The model name is defined in `get_llm()` in `graph/nodes.py` — change it there to switch models globally.

### Adding knowledge to VectorDB

Either paste text into the Gradio sidebar at runtime, or extend `data/sample_docs.txt` and delete `chroma_db/` to re-seed on next startup.