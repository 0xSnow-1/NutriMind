import os
import uuid
import requests
import streamlit as st

API_URL = os.getenv("NUTRIMIND_API_URL", "http://localhost:8000")
API_KEY = os.getenv("API_KEY", "")

st.set_page_config(page_title="NutriMind", page_icon="🥗", layout="centered")

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

[data-testid="stAppViewContainer"] { background: #0e1117; color: #e0e0e0; }
[data-testid="stHeader"] { background: transparent; }

h1 {
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: -0.02em !important;
    color: #e8e8e8 !important;
}

.description {
    font-family: 'Inter', sans-serif;
    font-weight: 300;
    font-size: 0.9rem;
    color: #8b8fa3;
    letter-spacing: -0.01em;
    margin-top: -0.5rem;
}

[data-testid="stSidebar"] {
    background: #111318;
    border-right: 1px solid #1e2028;
}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] { color: #c0c0c0; }

.stChatInputContainer {
    background: #1a1c23 !important;
    border: 1px solid #282a33 !important;
    border-radius: 12px !important;
}
[data-testid="stChatInputTextArea"] { color: #e0e0e0 !important; }

[data-testid="stChatMessageContent"] { color: #d0d4dd !important; font-size: 0.95rem; line-height: 1.6; }
[data-testid="stChatMessage"][aria-label="user"] [data-testid="stChatMessageContent"] {
    background: #1a2e3f !important;
    color: #b8d4e8 !important;
}
[data-testid="stChatMessage"][aria-label="assistant"] [data-testid="stChatMessageContent"] {
    background: #1a1c23 !important;
}

footer { display: none; }
#MainMenu { visibility: hidden; }
.stAppDeployButton { display: none; }

.thread-badge {
    font-family: 'Inter', monospace;
    font-size: 0.75rem;
    color: #6b6f80;
    background: #1a1c23;
    padding: 0.2rem 0.5rem;
    border-radius: 6px;
    border: 1px solid #282a33;
}
</style>
""",
    unsafe_allow_html=True,
)


def new_thread():
    return uuid.uuid4().hex[:12]


if "thread_id" not in st.session_state:
    st.session_state.thread_id = new_thread()
if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("NutriMind")
st.markdown(
    '<p class="description">Multi-agent AI nutrition assistant — stateful memory, longitudinal analysis, and human-in-the-loop medical flagging</p>',
    unsafe_allow_html=True,
)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask about nutrition, log a meal, or get a meal plan..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                headers = {"X-API-Key": API_KEY} if API_KEY else {}
                resp = requests.post(
                    f"{API_URL}/chat",
                    json={
                        "message": prompt,
                        "thread_id": st.session_state.thread_id,
                    },
                    headers=headers,
                    timeout=90,
                )
                resp.raise_for_status()
                reply = resp.json()["response"]
            except requests.exceptions.ConnectionError:
                reply = "Backend not reachable. Run `uv run uvicorn agent.app:api --host 0.0.0.0 --port 8000`."
            except Exception as e:
                reply = f"Error: {e}"

        st.markdown(reply)
        st.session_state.messages.append({"role": "assistant", "content": reply})

with st.sidebar:
    st.image("nutrimind.png", width=36)
    st.markdown("### NutriMind")
    st.markdown(
        f"Thread: <span class='thread-badge'>{st.session_state.thread_id}</span>",
        unsafe_allow_html=True,
    )
    st.markdown(f"**Messages** `{len(st.session_state.messages)}`")

    st.divider()

    st.markdown("### Agents")
    st.markdown(
        "- **Memory Agent** — User profiles, meal history\n"
        "- **Nutrition RAG Agent** — Evidence-based Q&A via FAISS + USDA\n"
        "- **Planning Agent** — Adaptive meal plans, goal drift detection, LLM-as-judge eval\n"
        "- **Intake Agent** — Meal logging, running macros, deficiency detection\n"
        "- **Insight Agent** — 14-day pattern analysis, medical flagging, streak tracking"
    )

    st.divider()

    st.markdown("### Tech Stack")
    st.markdown("LangGraph · FastAPI · PostgreSQL · AWS Bedrock · FAISS · LangSmith")

    st.divider()

    st.markdown(
        "Built by [Ahmed Gamal](https://github.com/0xSnow-1) — AI Agent Systems Engineer"
    )

    if st.button("New conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.thread_id = new_thread()
        st.rerun()
