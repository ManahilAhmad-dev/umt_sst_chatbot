# src/app.py - UMT-SST Academic Advisor (Multimodal Expansion Paradigm)
import streamlit as st
import time
import tempfile
from config import Config
from rag import retrieve, format_context, ask_llm
# from face_engine import FacultyFaceEngine
from ocr_extractor import scan_image, extract_faculty_json, merge_into_database
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

# --- Initialize Biometric Core Cache in Persistent Session State ---
# This ensures intensive facial vector extractions occur exactly once on boot
# if 'face_engine' not in st.session_state:
#     with st.spinner("Initializing biometric infrastructure and loading vectors..."):
#         try:
#             st.session_state.face_engine = FacultyFaceEngine()
#         except Exception as e:
#             st.error(f"Critical Boot Failure: Facial recognition engine failed to load. {e}")

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
            file_name=filename,
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

# ======================================================================
#   Multimodal Bifurcated Presentation Layout Tier
# ======================================================================
tab1, tab2, tab3 = st.tabs(["💬 Academic Query (RAG)", "📷 Faculty Photo Search", "📋 Update Counselling Schedule"])

# ----------------------------------------------------------------------
# TAB 1: Existing Text-Based RAG Application Sandbox
# ----------------------------------------------------------------------
with tab1:
    st.markdown("### Academic Handbook Query System")
    
    question = st.text_area(
        "Ask your question:",
        placeholder="E.g., What are the rules for FYP? Prerequisites for ML?",
        height=90,
        key="rag_question_input"
    )

    if st.button("🚀 Ask", key="rag_ask_button"):
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

# ----------------------------------------------------------------------
# TAB 2: Multimodal Faculty Recognition Module
# ----------------------------------------------------------------------
with tab2:
    st.subheader("SST Faculty Visual Identification")
    st.markdown("Upload a clear headshot image of any SST instructor below to recognize them instantly[cite: 1].")
        
    # Standard file uploader widget configured exclusively for image byte streams
    uploaded_img = st.file_uploader("Upload instructor face picture...", type=['jpg', 'jpeg', 'png'])
        
    if uploaded_img is not None:
        # Render a visual processing confirmation stream back to the UI layout
        st.image(uploaded_img, caption="Target Image", width=240)
                
        # Fire off processing calculation metrics inside a protected visual spinner context
        # with st.spinner("Extracting facial topology landmarks and evaluating signatures..."):
        #
        #     # Pass stream pointer arrays directly to your cached memory lookup functions
        #     result = st.session_state.face_engine.identify_face(uploaded_img)
        #
        #     if result["status"] == "success":
        #         prof = result["data"]
        #         conf = result['confidence'] * 100
        #         st.success(f"Match Confirmed via Few-Shot Euclidean Matrix Analysis (Confidence: {conf:.1f}%)[cite: 1].")
        #
        #         # Inline markup wrapper injected directly using university hex tokens (#003366)
        #         st.markdown(f"""
        #         <div style='background-color: #f7faff; border-left: 5px solid #003366; padding: 20px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); margin-top: 10px;'>
        #             <h3 style='margin-top:0; color: #003366; margin-bottom: 5px;'>Instructor Identified:</h3>
        #             <h2 style='color: #333; margin-top: 0;'>{prof.get('name', 'Unknown Instructor')}</h2>
        #         </div>
        #         """, unsafe_allow_html=True)
        #
        #     elif result["status"] == "unknown":
        #         st.warning(f"⚠️ {result['message']}")
        #     else:
        #         st.error(f"❌ {result['message']}")
        st.info("Faculty photo recognition is currently disabled.")

# ----------------------------------------------------------------------
# TAB 3: OCR Schedule Ingestion — Admin Tool for Counselling Hours
# ----------------------------------------------------------------------
with tab3:
    st.subheader("Upload Counselling Hours Schedule Photo")
    st.markdown(
        "Take a photo of the printed schedule posted outside a faculty office, "
        "upload it here, and the system will extract counselling hours automatically "
        "using OCR. Once saved, the chatbot in Tab 1 can answer student queries about "
        "that teacher's availability."
    )

    # Session state for OCR results so re-runs don't redo extraction
    if "ocr_extracted" not in st.session_state:
        st.session_state.ocr_extracted = None
    if "ocr_raw_text" not in st.session_state:
        st.session_state.ocr_raw_text = None

    schedule_img = st.file_uploader(
        "Upload schedule photo (JPG / PNG):",
        type=["jpg", "jpeg", "png"],
        key="schedule_upload"
    )

    if schedule_img is not None:
        st.image(schedule_img, caption="Uploaded Schedule", width=420)

        col_run, col_clear = st.columns([1, 4])
        run_ocr = col_run.button("🔍 Run OCR", key="run_ocr_btn")
        if col_clear.button("Reset", key="reset_ocr_btn"):
            st.session_state.ocr_extracted = None
            st.session_state.ocr_raw_text = None
            st.rerun()

        if run_ocr:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                tmp.write(schedule_img.read())
                tmp_path = tmp.name

            with st.spinner("Running OCR — this may take 20-40 s on first run (model download)…"):
                try:
                    raw_text = scan_image(tmp_path)
                    st.session_state.ocr_raw_text = raw_text
                except Exception as e:
                    st.error(f"OCR failed: {e}")
                    raw_text = None

            if raw_text:
                with st.spinner("Structuring data with LLM…"):
                    try:
                        record = extract_faculty_json(raw_text)
                        st.session_state.ocr_extracted = record
                    except Exception as e:
                        st.error(f"LLM structuring failed: {e}")

            try:
                os.unlink(tmp_path)
            except Exception:
                pass

        # ── Show raw OCR text in expander ──────────────────────────────
        if st.session_state.ocr_raw_text:
            with st.expander("🔎 Raw OCR Text (for debugging)"):
                st.text(st.session_state.ocr_raw_text)

        # ── Show editable extracted record ─────────────────────────────
        if st.session_state.ocr_extracted:
            rec = st.session_state.ocr_extracted
            st.markdown("---")
            st.markdown("#### Extracted Record — Review & Edit Before Saving")
            st.caption("Correct any OCR mistakes below, then click Save.")

            col1, col2 = st.columns(2)
            with col1:
                rec["name"]            = st.text_input("Full Name",            value=rec.get("name") or "")
                rec["title"]           = st.text_input("Designation",          value=rec.get("title") or "")
                rec["department"]      = st.text_input("Department",           value=rec.get("department") or "")
                rec["email"]           = st.text_input("Email",                value=rec.get("email") or "")
            with col2:
                rec["phone"]           = st.text_input("Phone",                value=rec.get("phone") or "")
                rec["office_location"] = st.text_input("Room / Office",        value=rec.get("office_location") or "")
                rec["office_timings"]  = st.text_input(
                    "Counselling Hours",
                    value=rec.get("office_timings") or "",
                    help="e.g.  Mon: 11:00-13:00, Tue: 11:00-13:00, Wed: 11:00-13:00"
                )
                rec["slug"]            = st.text_input(
                    "Slug (DB key)",
                    value=rec.get("slug") or "",
                    help="Lowercase first_last, e.g. shaista_habib"
                )

            st.session_state.ocr_extracted = rec

            st.markdown("---")
            if st.button("💾 Save to Faculty Database", key="save_ocr_btn"):
                record_copy = dict(st.session_state.ocr_extracted)
                if not record_copy.get("slug"):
                    st.error("Slug is required. Fill it in before saving.")
                elif not record_copy.get("name"):
                    st.error("Faculty name is required.")
                else:
                    try:
                        merge_into_database(record_copy, Config.FACULTY_JSON_PATH)
                        st.success(
                            f"Saved **{record_copy.get('name')}** to the database. "
                            "The chatbot can now answer counselling hour queries for this teacher."
                        )
                        st.session_state.ocr_extracted = None
                        st.session_state.ocr_raw_text = None
                    except Exception as e:
                        st.error(f"Save failed: {e}")

    else:
        st.info("No image uploaded yet. Take a clear, straight-on photo of the door schedule for best OCR accuracy.")