import json, pathlib

SCHEDULES = pathlib.Path("../data/faculty_schedules.json")
FACULTY_DB = pathlib.Path("../data/faculty_images/faculty_details.json")
SKIP = {"Ms. Fatima Shahzad", "Khaqan Zaheer"}
COUNSELLING_KEYWORDS = {"counseling", "counselling", "office hours"}

def get_counselling(schedule):
    slots = []
    for day, entries in schedule.items():
        if not isinstance(entries, list): continue
        for e in entries:
            if any(k in e.get("activity","").lower() for k in COUNSELLING_KEYWORDS):
                slots.append(f"{day[:3]}: {e['time']}")
    return " | ".join(slots) if slots else "N/A"

def make_slug(name):
    n = name.lower()
    for p in ["mr. ","ms. ","dr. m. ","dr. "]:
        n = n.replace(p, "")
    return n.strip().replace(" ","_").replace("-","_").replace(".","")

with open(SCHEDULES) as f: records = json.load(f)
with open(FACULTY_DB) as f: db = json.load(f)

added = updated = 0
for r in records:
    inst = r["instructor"]
    if inst["name"] in SKIP: continue
    key = make_slug(inst["name"])
    entry = {
        "name": inst["name"],
        "title": inst.get("designation","N/A"),
        "department": r.get("department","").replace("Department of ",""),
        "email": inst.get("email","N/A"),
        "phone": inst.get("contact_no","N/A"),
        "office_location": inst.get("room_no","N/A"),
        "office_timings": get_counselling(r.get("schedule",{}))
    }
    if key in db:
        db[key].update({k:v for k,v in entry.items() if v not in ("N/A","Not Specified","NA","")})
        updated += 1
    else:
        db[key] = entry
        added += 1

with open(FACULTY_DB,"w") as f: json.dump(db, f, indent=2, ensure_ascii=False)
print(f"✅ Done — {updated} updated, {added} new records added.")
