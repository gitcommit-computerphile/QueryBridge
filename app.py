import os
import uuid
import sqlite3
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

import gradio as gr
from langchain_core.messages import HumanMessage, AIMessage

from graph.graph import build_graph, SQLITE_DB_PATH
from vectordb.setup import add_documents_to_vectorstore

# ---------------------------------------------------------------------------
# Build graph once at startup
# ---------------------------------------------------------------------------
print("Initialising graph and vector store...")
graph, vectorstore = build_graph()
print("Ready.")


# ---------------------------------------------------------------------------
# Core chat function
# ---------------------------------------------------------------------------

def chat(message: str, history: list, thread_id: str):
    if not thread_id:
        thread_id = str(uuid.uuid4())

    config = {"configurable": {"thread_id": thread_id}}

    state_input = {"messages": [HumanMessage(content=message)]}
    result = graph.invoke(state_input, config=config)

    # last message in state is the AI response
    ai_messages = [
        m for m in result["messages"] if hasattr(m, "content") and not isinstance(m, HumanMessage)
    ]
    answer = ai_messages[-1].content if ai_messages else "Sorry, I could not generate a response."

    route_used = result.get("route", "unknown").upper()
    answer_with_badge = f"**[{route_used}]** {answer}"

    return answer_with_badge, thread_id


# ---------------------------------------------------------------------------
# Add-documents function (sidebar)
# ---------------------------------------------------------------------------

def add_docs(text: str):
    if not text.strip():
        return "Please enter some text to add."
    count = add_documents_to_vectorstore([text], vectorstore)
    return f"Added {count} chunk(s) to the knowledge base."


# ---------------------------------------------------------------------------
# Session helpers
# ---------------------------------------------------------------------------

def list_sessions() -> list[str]:
    """Return all thread_ids from the checkpoint DB, newest first."""
    if not Path(SQLITE_DB_PATH).exists():
        return []
    try:
        conn = sqlite3.connect(SQLITE_DB_PATH)
        cur = conn.execute(
            "SELECT DISTINCT thread_id FROM checkpoints ORDER BY rowid DESC"
        )
        rows = [r[0] for r in cur.fetchall()]
        conn.close()
        return rows
    except Exception:
        return []


def load_session_history(thread_id: str) -> list[dict]:
    """Reconstruct Gradio chat history from a stored LangGraph thread."""
    if not thread_id:
        return []
    try:
        config = {"configurable": {"thread_id": thread_id}}
        state = graph.get_state(config)
        messages = state.values.get("messages", [])
        history = []
        for m in messages:
            if isinstance(m, HumanMessage):
                history.append({"role": "user", "content": m.content})
            elif isinstance(m, AIMessage):
                history.append({"role": "assistant", "content": m.content})
        return history
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

def respond(message, history, thread_id_state):
    answer, thread_id = chat(message, history, thread_id_state)
    return answer, thread_id


with gr.Blocks(title="LangGraph RAG Router", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # LangGraph RAG Router
        Ask anything. The router will automatically decide whether to fetch the answer
        from **Wikipedia** (general knowledge) or the **Vector DB** (AI/ML knowledge base).

        > Conversations are persisted — your history survives restarts via the same Session ID.
        """
    )

    thread_id_state = gr.State(value=lambda: str(uuid.uuid4()))

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                label="Chat",
                height=520,
            )
            with gr.Row():
                msg_box = gr.Textbox(
                    placeholder="Ask me anything...",
                    show_label=False,
                    scale=5,
                    container=False,
                )
                send_btn = gr.Button("Send", variant="primary", scale=1)

        with gr.Column(scale=1):
            gr.Markdown("### Current Session")
            session_display = gr.Textbox(
                label="Session ID",
                interactive=False,
            )
            new_session_btn = gr.Button("New Session")

            gr.Markdown("### Previous Sessions")
            session_list = gr.Radio(
                label="Click to restore",
                choices=list_sessions(),
                interactive=True,
            )
            refresh_btn = gr.Button("Refresh List")

            gr.Markdown("### Add to Knowledge Base")
            doc_input = gr.Textbox(
                label="Paste text",
                lines=6,
                placeholder="Paste any text to add it to the vector DB...",
            )
            add_btn = gr.Button("Add to VectorDB")
            add_status = gr.Textbox(label="Status", interactive=False)

    # ── wiring ──────────────────────────────────────────────────────────────

    def submit_message(message, history, thread_id):
        if not message.strip():
            return history, "", thread_id, gr.update()
        answer, thread_id = chat(message, history, thread_id)
        history = history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": answer},
        ]
        return history, "", thread_id, gr.update(choices=list_sessions())

    send_btn.click(
        submit_message,
        inputs=[msg_box, chatbot, thread_id_state],
        outputs=[chatbot, msg_box, thread_id_state, session_list],
    )
    msg_box.submit(
        submit_message,
        inputs=[msg_box, chatbot, thread_id_state],
        outputs=[chatbot, msg_box, thread_id_state, session_list],
    )

    # keep session display in sync with state
    thread_id_state.change(
        lambda tid: tid,
        inputs=[thread_id_state],
        outputs=[session_display],
    )

    # restore a previous session from the radio list
    def restore_from_list(selected_tid):
        if not selected_tid:
            return gr.update(), gr.update(), gr.update()
        history = load_session_history(selected_tid)
        return selected_tid, history, selected_tid

    session_list.change(
        restore_from_list,
        inputs=[session_list],
        outputs=[thread_id_state, chatbot, session_display],
    )

    # start a brand new session
    def new_session():
        tid = str(uuid.uuid4())
        return tid, [], None

    new_session_btn.click(
        new_session,
        outputs=[thread_id_state, chatbot, session_list],
    )

    # refresh the session list
    refresh_btn.click(
        lambda: gr.update(choices=list_sessions()),
        outputs=[session_list],
    )

    add_btn.click(
        add_docs,
        inputs=[doc_input],
        outputs=[add_status],
    )

    # initialise on load
    demo.load(
        lambda tid: (tid, gr.update(choices=list_sessions())),
        inputs=[thread_id_state],
        outputs=[session_display, session_list],
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
