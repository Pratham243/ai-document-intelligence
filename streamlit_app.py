import os
import sys
import time
import subprocess
import signal
import requests
import streamlit as st

# --- Spawn FastAPI backend as subprocess ---
@st.cache_resource
def start_backend():
    env = os.environ.copy()
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "8000"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    # Wait for backend to be ready
    for _ in range(30):
        try:
            r = requests.get("http://127.0.0.1:8000/api/v1/health", timeout=2)
            if r.status_code == 200:
                break
        except Exception:
            pass
        time.sleep(1)
    return proc

API_BASE = "http://127.0.0.1:8000/api/v1"

# Start backend
_backend = start_backend()

# --- Page config ---
st.set_page_config(
    page_title="AI Document Intelligence",
    page_icon="📄",
    layout="wide",
)

st.title("📄 AI Document Intelligence")
st.caption("Production RAG pipeline — hybrid search with BM25 + semantic retrieval")

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Settings")

    llm_mode = st.radio(
        "LLM Mode",
        ["groq", "ollama", "none"],
        captions=[
            "Groq Llama 3.3 70B (cloud)",
            "Ollama local (GDPR mode)",
            "No LLM (raw retrieval)",
        ],
        index=0,
    )

    st.divider()
    st.header("📁 Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF, DOCX, or TXT files",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        for uf in uploaded_files:
            with st.spinner(f"Ingesting {uf.name}..."):
                try:
                    files = {"file": (uf.name, uf.getvalue(), "application/octet-stream")}
                    resp = requests.post(f"{API_BASE}/documents", files=files, timeout=120)
                    if resp.status_code == 200:
                        data = resp.json()
                        st.success(f"✅ {uf.name}: {data['num_chunks']} chunks indexed")
                    else:
                        st.error(f"❌ {uf.name}: {resp.text}")
                except Exception as e:
                    st.error(f"❌ {uf.name}: {str(e)}")

    st.divider()
    st.header("📚 Indexed Documents")
    if st.button("Refresh document list"):
        st.rerun()

    try:
        docs_resp = requests.get(f"{API_BASE}/documents", timeout=10)
        if docs_resp.status_code == 200:
            docs = docs_resp.json()
            if docs:
                for doc in docs:
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.text(f"📄 {doc['filename']} ({doc['num_chunks']} chunks)")
                    with col2:
                        if st.button("🗑️", key=f"del_{doc['doc_id']}"):
                            requests.delete(f"{API_BASE}/documents/{doc['doc_id']}", timeout=10)
                            st.rerun()
            else:
                st.info("No documents indexed yet.")
    except Exception:
        st.warning("Backend not ready yet.")

    st.divider()
    st.header("📊 Evaluation")
    try:
        eval_resp = requests.get(f"{API_BASE}/evaluation/results", timeout=10)
        if eval_resp.status_code == 200:
            results = eval_resp.json()
            st.json(results.get("summary", {}))
        else:
            st.info("No evaluation results. Run: `pytest tests/evaluation/`")
    except Exception:
        st.info("Evaluation results unavailable.")

# --- Chat interface ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📎 Sources"):
                for src in msg["sources"]:
                    st.markdown(f"**{src['filename']}** (chunk {src['chunk_index']}, score: {src['score']:.3f})")
                    st.text(src["text"])
        if msg.get("trace_url"):
            st.caption(f"🔍 [LangFuse Trace]({msg['trace_url']})")

# Voice input
st.markdown("---")
col_voice, col_text = st.columns([1, 4])
with col_voice:
    audio_data = st.audio_input("🎤 Voice input")

voice_question = None
if audio_data is not None:
    with st.spinner("Transcribing..."):
        try:
            files = {"audio": ("recording.wav", audio_data.getvalue(), "audio/wav")}
            data = {"mode": llm_mode}
            resp = requests.post(f"{API_BASE}/query/voice", files=files, data=data, timeout=120)
            if resp.status_code == 200:
                result = resp.json()
                voice_question = result.get("transcription", "")
                st.session_state.messages.append({"role": "user", "content": f"🎤 {voice_question}"})
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": result.get("sources", []),
                    "trace_url": result.get("trace_url"),
                })
                st.rerun()
            else:
                st.error(f"Voice query failed: {resp.text}")
        except Exception as e:
            st.error(f"Voice query error: {str(e)}")

# Text input
if prompt := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching and generating answer..."):
            try:
                resp = requests.post(
                    f"{API_BASE}/query",
                    json={"question": prompt, "mode": llm_mode},
                    timeout=120,
                )
                if resp.status_code == 200:
                    result = resp.json()
                    st.markdown(result["answer"])

                    sources = result.get("sources", [])
                    if sources:
                        with st.expander("📎 Sources"):
                            for src in sources:
                                st.markdown(f"**{src['filename']}** (chunk {src['chunk_index']}, score: {src['score']:.3f})")
                                st.text(src["text"])

                    trace_url = result.get("trace_url")
                    if trace_url:
                        st.caption(f"🔍 [LangFuse Trace]({trace_url})")

                    st.caption(f"⚡ {result['latency_ms']:.0f}ms | {result['provider']}/{result['model']}")

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": result["answer"],
                        "sources": sources,
                        "trace_url": trace_url,
                    })
                elif resp.status_code == 400:
                    error_detail = resp.json().get("detail", resp.text)
                    st.error(f"⚠️ {error_detail}")
                    st.session_state.messages.append({"role": "assistant", "content": f"⚠️ {error_detail}"})
                else:
                    st.error(f"Error: {resp.text}")
            except Exception as e:
                st.error(f"Error: {str(e)}")
