"""Scaffold a new experiment folder from the template and append a draft registry row.

Usage:  python scripts/new_experiment.py "<slug>" ["Title"]
"""
import datetime as dt
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = ROOT / "templates" / "experiment_template"


def main():
    if len(sys.argv) < 2:
        print("usage: new_experiment.py <slug> [Title]")
        sys.exit(1)
    slug = sys.argv[1].strip().lower().replace(" ", "-")
    title = sys.argv[2] if len(sys.argv) > 2 else slug.replace("-", " ").title()
    date = dt.date.today().isoformat()
    exp_id = f"{date}_{slug}"
    dest = ROOT / "experiments" / exp_id
    if dest.exists():
        print(f"already exists: {dest}")
        sys.exit(1)

    shutil.copytree(TEMPLATE, dest)
    # substitute placeholders
    for p in dest.rglob("*"):
        if p.is_file():
            txt = p.read_text()
            txt = txt.replace("DATE_SLUG", exp_id).replace("DATE", date).replace("TITLE", title)
            p.write_text(txt)

    row = {
        "id": exp_id, "date": date, "title": title, "slug": slug,
        "hypothesis": "", "status": "in_progress",
        "novelty_check": {"verdict": "unchecked", "sources": []},
        "links": {"folder": f"experiments/{exp_id}"}, "tags": [],
    }
    with open(ROOT / "registry.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")

    print(f"created {dest}")
    print("next: edit experiment.yaml + run.py, run novelty_check.py, then run.py")


if __name__ == "__main__":
    main()
