"""
ocr_extractor.py
----------------
Step 1 : EasyOCR  → raw text from a door-schedule photo
Step 2 : Groq LLM → structured JSON block for faculty_details.json
"""

import os
import json
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

# ─── Environment ────────────────────────────────────────────────────────────
load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),       # holds the gsk_... Groq key
    base_url=os.getenv("OPENAI_BASE_URL"),     # https://api.groq.com/openai/v1
)
MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")


# ─── Step 1 : OCR ────────────────────────────────────────────────────────────
def scan_image(image_path: str) -> str:
    """Run EasyOCR on an image and return a single joined string."""
    import easyocr  # type: ignore[import-untyped]  # lazy import — heavy model load, only when needed

    print("🤖 Loading OCR model (first run downloads ~100 MB)…")
    reader = easyocr.Reader(["en"], gpu=True)

    print(f"📸 Scanning {image_path} …")
    results = reader.readtext(image_path)

    lines = []
    for (_bbox, text, conf) in results:
        indicator = "✓" if conf >= 0.5 else "?"
        print(f"  {indicator} [{conf:.2f}]  {text}")
        lines.append(text)

    return "\n".join(lines)


# ─── Step 2 : LLM → structured JSON ─────────────────────────────────────────
SYSTEM_PROMPT = """You are a data-entry assistant for a university faculty database at UMT \
(University of Management and Technology).

You receive raw OCR text scanned from a printed "Counselling Hours / Instructor Time Table" \
sheet posted outside a faculty member's office. The sheet has:
1. A header block with faculty name, designation, department, contact, email, and room number.
2. A timetable grid where:
   - Rows are days (Saturday, Sunday, Monday, Tuesday, Wednesday, Thursday, Friday)
   - Columns are time slots (08:00-09:00, 09:00-10:00, 10:00-11:00, 11:00-12:00, 12:00-13:00, \
13:00-14:00, 14:00-15:00, 15:00-16:00, 16:00-17:00)
   - Cells contain "Counselling Hours" (available for students), a course name (teaching), \
"OFFDAY", or are empty.

Your task:
- Extract the faculty header info.
- Identify ALL time slots marked "Counselling Hours" for each day and consolidate consecutive \
slots into ranges. Skip OFFDAY and Sunday/Saturday rows.
- Express counselling hours compactly, e.g. "Mon: 11:00-13:00, Tue: 11:00-13:00, \
Wed: 11:00-13:00, Thu: 11:00-13:00, Fri: 11:00-13:00"
- For slug, use lowercase first_last name (e.g. "shaista_habib" for "Dr. Shaista Habib").

Return ONLY one valid JSON object — no markdown fences, no explanation, no extra keys:
{
  "slug":            "<lowercase_first_last>",
  "name":            "<Full Name as printed, including Dr./Mr./Ms. if present>",
  "title":           "<Lecturer / Assistant Professor / Associate Professor / Professor>",
  "department":      "<department name>",
  "email":           "<email or null>",
  "phone":           "<phone number or null>",
  "office_location": "<room code, e.g. CB1-507-13 or null>",
  "office_timings":  "<counselling schedule string or null>"
}"""


def _ocr_with_confidence(image_path: str):
    """
    Same OCR pass as scan_image() but also returns the raw EasyOCR result list
    so callers can inspect per-line confidence without running the model twice.
    Returns (text: str, results: list[tuple[bbox, text, conf]])
    """
    import easyocr  # type: ignore[import-untyped]

    print("🤖 Loading OCR model (first run downloads ~100 MB)…")
    reader = easyocr.Reader(["en"], gpu=True)

    print(f"📸 Scanning {image_path} …")
    results = reader.readtext(image_path)

    lines = []
    for (_bbox, text, conf) in results:
        indicator = "✓" if conf >= 0.5 else "?"
        print(f"  {indicator} [{conf:.2f}]  {text}")
        lines.append(text)

    return "\n".join(lines), results


# ─── Step 2b : Vision fallback — send image directly to GPT-4o ───────────────
VISION_SYSTEM_PROMPT = (
    "You are a data-entry assistant for a university faculty database. "
    "You are looking at a photo of a printed door schedule posted outside "
    "a faculty member's office at UMT (University of Management and Technology). "
    "Extract the faculty record and return a single valid JSON object with these keys: "
    "slug, name, title, department, email, phone, office_location, office_timings. "
    "Use null for any field you cannot find. office_timings should be a readable "
    "summary like 'Mon: 11:00-13:00, Wed: 14:00-16:00'. "
    "Use underscores not hyphens in the slug (e.g. basit_sattar not basit-sattar). "
    "Return ONLY the JSON object, no markdown, no explanation."
)


def extract_faculty_from_image_vision(image_path: str) -> dict:
    """Send the image directly to GPT-4o vision and parse the returned JSON."""
    import base64

    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")

    suffix = Path(image_path).suffix.lower().lstrip(".")
    mime = "image/jpeg" if suffix in ("jpg", "jpeg") else f"image/{suffix}"

    print("\n👁️  Sending image to GPT-4o vision…")
    response = client.chat.completions.create(
        model="gpt-4o",
        temperature=0.0,
        max_tokens=512,
        messages=[
            {"role": "system", "content": VISION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{b64}"},
                    }
                ],
            },
        ],
    )

    raw = (response.choices[0].message.content or "").strip()

    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.lower().startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"⚠️  JSON parse failed: {e}\nRaw output:\n{raw}")
        return {"raw_llm_output": raw}


def extract_faculty_json(ocr_text: str) -> dict:
    """Send OCR text to Groq LLM and parse the returned JSON."""
    print("\n🧠 Sending to Groq LLM for structuring…")

    response = client.chat.completions.create(
        model=MODEL,
        temperature=0.0,           # deterministic for data extraction
        max_tokens=512,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": f"OCR TEXT:\n{ocr_text}"},
        ],
    )

    raw = (response.choices[0].message.content or "").strip()

    # Strip accidental markdown fences
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.lower().startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    try:
        record = json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"⚠️  JSON parse failed: {e}")
        print("Raw LLM output:\n", raw)
        record = {"raw_llm_output": raw}

    return record


# ─── Step 3 : Merge helper (optional convenience) ────────────────────────────
def merge_into_database(new_record: dict, db_path: str) -> None:
    """
    Upsert new_record into faculty_details.json using 'slug' as the key.
    Creates the file if it does not exist yet.
    """
    db_file = Path(db_path)
    db = {}
    if db_file.exists():
        with open(db_file, encoding="utf-8") as f:
            db = json.load(f)

    slug = new_record.pop("slug", None)
    if not slug:
        print("⚠️  No slug found — skipping merge. Record:", new_record)
        return

    db[slug] = new_record
    with open(db_file, "w", encoding="utf-8") as f:
        json.dump(db, f, indent=2, ensure_ascii=False)

    print(f"✅  Saved '{slug}' → {db_file}")


# ─── CLI entry-point ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    db_path = sys.argv[1] if len(sys.argv) > 1 else str(
        Path(__file__).parent.parent / "data" / "faculty_images" / "faculty_details.json"
    )

    schedules_dir = Path(__file__).parent.parent / "data" / "schedules"
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    images = sorted(
        p for p in schedules_dir.iterdir()
        if p.is_file() and p.suffix.lower() in image_exts
    )

    if not images:
        print(f"❌  No image files found in {schedules_dir}")
        sys.exit(1)

    print(f"\n📂  Found {len(images)} image(s) in {schedules_dir}\n")

    saved_count = 0

    for idx, img_path in enumerate(images, 1):
        print(f"\n{'='*60}")
        print(f"Processing {idx}/{len(images)}: {img_path.name}")
        print("=" * 60)

        # --- OCR + routing ---
        try:
            raw_text, ocr_results = _ocr_with_confidence(str(img_path))
        except Exception as e:
            print(f"⚠️  OCR failed for {img_path.name}: {e} — skipping.")
            continue

        print("\n📄 Raw OCR output:\n" + "-" * 40)
        print(raw_text)
        print("-" * 40)

        high_conf_count = sum(1 for (_bbox, _text, conf) in ocr_results if conf >= 0.5)

        try:
            if high_conf_count < 3:
                print(f"\n🔍 Using Vision mode  (only {high_conf_count} high-confidence OCR line(s))")
                record = extract_faculty_from_image_vision(str(img_path))
            else:
                print(f"\n📝 Using OCR+LLM mode  ({high_conf_count} high-confidence OCR lines)")
                record = extract_faculty_json(raw_text)
        except Exception as e:
            print(f"⚠️  Extraction failed for {img_path.name}: {e} — skipping.")
            continue

        print("\n📦 Structured record:")
        print(json.dumps(record, indent=2, ensure_ascii=False))

        # --- Validity check ---
        name = record.get("name")
        if not name or name == "null":
            print(f"⚠️  Could not extract faculty name from {img_path.name} — skipping.")
            continue

        # --- Save prompt ---
        try:
            answer = input("\n💾  Save this? [y/N] ").strip().lower()
        except EOFError:
            answer = "n"

        if answer == "y":
            merge_into_database(record, db_path)
            saved_count += 1
        else:
            print("Skipped.")

    print(f"\n✅  Done. {saved_count} record(s) saved.")
