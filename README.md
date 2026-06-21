# UMT-SST Academic Advisor

A Streamlit-based chatbot for the UMT School of Systems and Technology (Department of Artificial Intelligence). It answers student questions about the academic handbook (RAG), helps identify faculty from photos, and lets admins update faculty counselling-hours data by uploading a photo of the schedule posted outside their office.

## Features

| Tab | What it does |
|---|---|
| **Academic Query (RAG)** | Students ask questions about the academic handbook (FYP rules, course prerequisites, etc.) and get answers grounded in the handbook PDF via a FAISS vector store. |
| **Faculty Photo Search** | Looks up faculty contact info, office location, and counselling hours by name or department. |
| **Update Counselling Schedule** | Admin uploads a photo of a door schedule; the system extracts the faculty member's name, contact details, and counselling hours automatically and saves them to the faculty database. |

> Face-recognition ("identify faculty from a live photo") is present in the codebase (`face_engine.py`) but currently disabled in the UI — see [Known Issues](#known-issues).

## Architecture

```
Student question
      │
      ▼
┌─────────────┐      ┌──────────────────┐
│   app.py    │─────▶│     rag.py       │──▶ FAISS vector store ──▶ Academic Handbook PDF
│ (Streamlit) │      └──────────────────┘
│             │      ┌──────────────────┐
│             │─────▶│ faculty_details  │──▶ name / counselling-hours lookup
│             │      │     .json        │
└─────────────┘      └──────────────────┘
      ▲
      │  (admin uploads door-schedule photo)
      │
┌─────────────────────────────────────────┐
│            ocr_extractor.py              │
│                                           │
│  EasyOCR scan ──▶ confidence check        │
│       │                                  │
│       ├─ ≥3 high-confidence lines        │
│       │    └─▶ Groq/OpenAI text LLM      │
│       │                                  │
│       └─ <3 high-confidence lines        │
│            └─▶ GPT-4o Vision (reads      │
│                 the photo directly)      │
│                                           │
│  Either path → structured JSON           │
│       └─▶ merge_into_database()          │
│            └─▶ faculty_details.json      │
└─────────────────────────────────────────┘
```

## Project Structure

```
umt_sst_chatbot/
├── data/
│   ├── faculty_images/
│   │   ├── faculty_details.json     # faculty registry (name, email, office, counselling hours)
│   │   └── *.png                    # faculty photos for face recognition
│   ├── faculty_schedules.json       # raw structured schedule extractions (one entry per door photo)
│   └── schedules/                   # door-schedule photos awaiting OCR processing
├── src/
│   ├── app.py                       # Streamlit UI — all 3 tabs
│   ├── rag.py                       # FAISS retrieval + LLM answer generation
│   ├── ocr_extractor.py             # EasyOCR + GPT-4o Vision schedule extraction pipeline
│   ├── build_faculty_db.py          # one-off script: merges faculty_schedules.json into faculty_details.json
│   ├── face_engine.py               # face_recognition wrapper (currently disabled in app.py)
│   ├── ingest.py                    # builds the FAISS index from the handbook PDF
│   └── config.py                    # environment / path configuration
├── .env                             # API keys (gitignored)
├── requirements.txt
└── pyrightconfig.json
```

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

`face_recognition` requires `dlib`, which can be tricky on Windows. If `pip install` fails for it:

```bash
conda install -c conda-forge dlib
pip install face_recognition
```

### 2. Environment variables

Create a `.env` file in the project root:

```
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-4o-mini
EMBEDDING_MODEL=all-MiniLM-L6-v2
```

> The pipeline also supports Groq — set `OPENAI_API_KEY` to a `gsk_...` key and `OPENAI_BASE_URL` to `https://api.groq.com/openai/v1` instead.

### 3. GPU acceleration (optional, recommended)

EasyOCR defaults to CPU. To use an NVIDIA GPU:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

Then verify:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

### 4. Run the app

```bash
cd src
python -m streamlit run app.py
```

## Updating Faculty Counselling Hours

**Option A — via the app (recommended for one-off updates):**
Use the "Update Counselling Schedule" tab to upload a door-schedule photo. The OCR/Vision pipeline extracts the data and lets you review/edit before saving.

**Option B — batch processing (recommended for bulk uploads):**
1. Drop all door-schedule photos into `data/schedules/`
2. Run the CLI pipeline from `src/`:
   ```bash
   python ocr_extractor.py
   ```
3. It automatically routes each image to OCR+LLM or GPT-4o Vision depending on image quality, and prompts before saving each record.

## How the OCR/Vision Routing Works

Door schedules are often blurry, angled, or poorly lit phone photos. Rather than relying on OCR alone:

1. EasyOCR scans the image and returns per-line confidence scores.
2. If **3 or more lines** have confidence ≥ 0.5, the raw OCR text is sent to a text LLM for structuring (cheaper, faster).
3. If **fewer than 3 lines** are confident, the image is sent directly to **GPT-4o Vision**, which reads the photo itself rather than relying on noisy OCR text.

This handles the full range of photo quality without requiring users to retake pictures.

## Known Issues

- **Face recognition UI disabled** — `face_engine.py` and its Streamlit tab integration are commented out in `app.py` pending `dlib`/`face_recognition` installation issues on the target machine. Re-enable by uncommenting the import and the relevant block in the "Faculty Photo Search" tab.
- **3D graph visualization** (if used elsewhere in the broader project) has a blank-screen issue tied to CDN dependencies — kept local-only as a workaround.
- **Incomplete faculty schedules** — the following faculty still have `office_timings: "N/A"` pending a clear door-schedule photo: `ahmad_dawood`, `ahmad_yar`, `dr_ashfaq_ahmad`, `hafiza_hina_javed`, `jawad_hassan`, `jawairia_rasheed`, `mehak_saleem`, `noreen_ashraf`, `reshmail_fatima`, `sardar_waqar_khan`, `mahmood_hussain`.

## Deployment Notes

- Cannot host on Vercel — this is a Python server app (Streamlit), not a static site.
- Large `.json`/`.pkl` files (vector index, faculty photo embeddings) exceed GitHub/Vercel free-tier limits — host code on GitHub, large data files on Google Drive, and document the download step for anyone setting up the project locally.

## Security

`.env` is gitignored. If an API key is ever accidentally committed or shared, rotate it immediately at your provider's dashboard (OpenAI or Groq).
