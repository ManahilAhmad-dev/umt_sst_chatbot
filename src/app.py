# src/app.py - UMT-SST Academic Advisor (ChatGPT-style UI with Sidebar History)
import streamlit as st
import time
from config import Config
from rag import retrieve, format_context, ask_llm
from dotenv import load_dotenv
import os
import pathlib
import base64
import json

# --- Load Environment Variables ---
load_dotenv()
project_root = os.path.dirname(os.path.dirname(__file__))
env_path = os.path.join(project_root, ".env")
load_dotenv(dotenv_path=env_path)

# --- Logo Path ---
current_dir = pathlib.Path(__file__).parent
LOGO_PATH = current_dir / "assets" / "download.png"

with open(LOGO_PATH, "rb") as img_file:
    base64_logo = base64.b64encode(img_file.read()).decode()

# --- Streamlit Page Config ---
st.set_page_config(
    page_title="UMT-SST Academic Advisor",
    page_icon=str(LOGO_PATH),
    layout="wide"
)
# ===============================
#   Custom ChatGPT-Style CSS
# ===============================
st.markdown(
    f"""
    <style>
        :root {{
            --umt-blue: #003366;
            --umt-light-blue: #e6f0ff;
            --white: #ffffff;
        }}

        /* Main background */
        .main {{
            background-color: #f7faff !important;
        }}

        /* Chat bubbles */
        .user-msg {{
            background: var(--umt-blue);
            color: white;
            padding: 15px;
            border-radius: 12px;
            margin-bottom: 12px;
            max-width: 80%;
        }}

        .bot-msg {{
            background: var(--white);
            color: #222;
            padding: 15px;
            border-radius: 12px;
            border: 1px solid #bcd0ff;
            margin-bottom: 12px;
            max-width: 80%;
        }}

        /* Title */
        .title {{
            text-align: center;
            font-size: 42px;
            font-weight: 900;
            color: var(--umt-blue);
            margin-bottom: -10px;
        }}

        /* Subtitle */
        .subtitle {{
            text-align: center;
            font-size: 20px;
            color: #333;
            margin-bottom: 25px;
        }}

        /* Input Box */
        textarea {{
            border-radius: 12px !important;
            border: 2px solid var(--umt-blue) !important;
        }}

        /* Buttons */
        .stButton>button {{
            background-color: var(--umt-blue);
            color: white;
            padding: 8px 20px;
            font-size: 18px;
            border-radius: 8px;
            border: none;
        }}
        .stButton>button:hover {{
            background-color: #002244;
        }}

        /* Sidebar styling */
        section[data-testid="stSidebar"] {{
            background-color: var(--umt-blue);
        }}
        .sidebar-title {{
            color: white;
            font-size: 26px;
            font-weight: 700;
            text-align: center;
            padding-bottom: 10px;
        }}
        .history-item {{
            background: #ffffff22;
            padding: 12px;
            border-radius: 8px;
            margin-bottom: 10px;
            color: white;
            border-left: 4px solid #9ec9ff;
        }}
        .history-item:hover {{
            background: #ffffff33;
            cursor: pointer;
        }}
    </style>
    """,
    unsafe_allow_html=True
)

# ===============================
#        Left Sidebar
# ===============================
st.sidebar.markdown(f"<div class='sidebar-title'>📘 Chat History</div>", unsafe_allow_html=True)

if "history" not in st.session_state:
    st.session_state.history = []

# Show history in sidebar
if st.session_state.history:
    for i, item in enumerate(reversed(st.session_state.history[-30:])):
        st.sidebar.markdown(
            f"""
            <div class='history-item'>
                <b>{item['timestamp'][11:]}</b><br>
                {item['question'][:40]}...
            </div>
            """,
            unsafe_allow_html=True
        )
else:
    st.sidebar.info("No conversations yet.")

# Clear History
if st.sidebar.button("🗑️ Clear History"):
    st.session_state.history = []
    st.rerun()

# Export History
if st.sidebar.button("⬇️ Export JSON"):
    if st.session_state.history:
        filename = f"umt_sst_chat_{int(time.time())}.json"
        json_data = json.dumps(st.session_state.history, indent=2)
        st.sidebar.download_button(
    "Download File",
    json_data,
    file_name=filename,   # ✅ CORRECT
    mime="application/json"
)

       
    else:
        st.sidebar.warning("No chat data available.")

# ===============================
#            HEADER
# ===============================
st.markdown(
    f"""
    <div style="text-align:center;">
        <img src="data:image/png;base64,{base64_logo}" 
             style="height:160px; margin-bottom:-10px;" />
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("<div class='title'>UMT-SST Academic Advisor</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Your intelligent companion for academic queries</div>", unsafe_allow_html=True)

# ===============================
#           Chat Input
# ===============================
question = st.text_area(
    "Ask your question:",
    placeholder="E.g., What are the rules for FYP? Prerequisites for ML?",
    height=90
)

if st.button("🚀 Ask"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("🔍 Searching knowledge base..."):
            chunks = retrieve(question, top_k=Config.TOP_K)

        if not chunks:
            st.error("No relevant documents found.")
        else:
            context = format_context(chunks)

            with st.spinner("🧠 Generating answer..."):
                answer = ask_llm(question, context)

            # Display chat bubbles
            st.markdown(f"<div class='user-msg'>🤔 <b>You:</b><br>{question}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='bot-msg'>💡 <b>UMT-SST Advisor:</b><br>{answer}</div>", unsafe_allow_html=True)

            # Save history
            st.session_state.history.append({
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "question": question,
                "answer": answer,
                "sources": [c['source'] for c in chunks]
            })

            # Expandable sources
            with st.expander("📚 Retrieved Sources"):
                for c in chunks:
                    preview = c["text"][:350] + ("..." if len(c["text"]) > 350 else "")
                    st.info(f"**Source:** {c['source']} — Chunk {c['chunk_id']}\n\n{preview}")
