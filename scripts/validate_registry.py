"""Validate registry.jsonl: valid JSON, required fields, unique ids, status enum.

Exit code 0 = ok, 1 = problems found.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REQUIRED = {"id", "date", "title", "slug", "hypothesis", "status"}
STATUSES = {"idea", "in_progress", "done", "failed", "abandoned", "spun_out"}


def main():
    path = ROOT / "registry.jsonl"
    if not path.exists():
        print("no registry.jsonl yet — nothing to validate")
        return
    errors, ids = [], set()
    for i, line in enumerate(path.read_text().splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as e:
            errors.append(f"line {i}: invalid JSON ({e})")
            continue
        missing = REQUIRED - row.keys()
        if missing:
            errors.append(f"line {i} ({row.get('id','?')}): missing {sorted(missing)}")
        if row.get("status") not in STATUSES:
            errors.append(f"line {i} ({row.get('id','?')}): bad status {row.get('status')!r}")
        rid = row.get("id")
        if rid in ids:
            errors.append(f"line {i}: duplicate id {rid}")
        ids.add(rid)

    if errors:
        print("REGISTRY INVALID:")
        for e in errors:
            print("  -", e)
        sys.exit(1)
    print(f"registry OK — {len(ids)} experiments")


if __name__ == "__main__":
    main()
