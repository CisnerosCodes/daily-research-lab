"""Regenerate the README index table from registry.jsonl (between INDEX markers)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
START, END = "<!-- INDEX:START -->", "<!-- INDEX:END -->"
BADGE = {"done": "✅", "in_progress": "🔷", "failed": "❌",
         "abandoned": "🗑️", "spun_out": "🚀", "idea": "💡"}


def main():
    reg = ROOT / "registry.jsonl"
    rows = []
    if reg.exists():
        for line in reg.read_text().splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    rows.sort(key=lambda r: r.get("date", ""), reverse=True)

    lines = [f"**{len(rows)} experiments logged.**", "",
             "| Date | Experiment | Status | Result |", "|---|---|---|---|"]
    for r in rows:
        folder = r.get("links", {}).get("folder", "")
        title = f"[{r.get('title','?')}]({folder})" if folder else r.get("title", "?")
        badge = BADGE.get(r.get("status"), "") + " " + r.get("status", "")
        summ = (r.get("result_summary", "") or "").replace("|", "\\|")[:90]
        lines.append(f"| {r.get('date','')} | {title} | {badge} | {summ} |")
    table = "\n".join(lines)

    readme = ROOT / "README.md"
    txt = readme.read_text()
    pre, rest = txt.split(START)
    _, post = rest.split(END)
    readme.write_text(f"{pre}{START}\n{table}\n{END}{post}")
    print(f"index rebuilt: {len(rows)} rows")


if __name__ == "__main__":
    main()
