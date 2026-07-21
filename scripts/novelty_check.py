"""Search-before-create: query arXiv + OpenAlex (and print GitHub/HF hints) for a topic.

Usage:  python scripts/novelty_check.py "looped transformer parity length generalization"

Stdlib-only (urllib). Papers with Code is intentionally NOT used (shut down July 2025).
Prints results and a JSON block you can paste into the experiment's registry row.
"""
import json
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

UA = {"User-Agent": "daily-research-lab/1.0 (novelty-check)"}


def _get(url):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read().decode("utf-8", "replace")


def arxiv(query, n=5):
    q = urllib.parse.quote(f'all:"{query}"')
    url = f"http://export.arxiv.org/api/query?search_query={q}&start=0&max_results={n}&sortBy=relevance"
    out = []
    try:
        root = ET.fromstring(_get(url))
        ns = {"a": "http://www.w3.org/2005/Atom"}
        for e in root.findall("a:entry", ns):
            out.append({"title": " ".join(e.find("a:title", ns).text.split()),
                        "url": e.find("a:id", ns).text})
    except Exception as e:
        out.append({"title": f"[arxiv error: {e}]", "url": ""})
    return out


def openalex(query, n=5):
    q = urllib.parse.quote(query)
    url = f"https://api.openalex.org/works?search={q}&per-page={n}&mailto=lab@example.com"
    out = []
    try:
        data = json.loads(_get(url))
        for w in data.get("results", [])[:n]:
            out.append({"title": w.get("display_name", "?"),
                        "url": w.get("id", ""),
                        "cited_by": w.get("cited_by_count", 0)})
    except Exception as e:
        out.append({"title": f"[openalex error: {e}]", "url": ""})
    return out


def main():
    if len(sys.argv) < 2:
        print('usage: novelty_check.py "<query>"')
        sys.exit(1)
    query = " ".join(sys.argv[1:])
    print(f"\n=== novelty check: {query!r} ===\n")

    print("arXiv:")
    a = arxiv(query)
    for r in a:
        print(f"  - {r['title']}\n    {r['url']}")
    time.sleep(3)  # arXiv asks for 3s between calls

    print("\nOpenAlex:")
    o = openalex(query)
    for r in o:
        print(f"  - {r['title']} (cited {r.get('cited_by','?')})\n    {r['url']}")

    print("\nGitHub (run manually):")
    print(f'  gh search repos "{query}" --language python --sort stars')
    print(f'  gh search code "{query}"')
    print("\nHugging Face (run manually):")
    print(f'  python -c \'from huggingface_hub import HfApi; print([m.id for m in HfApi().list_models(search="{query}", limit=10)])\'')

    block = {"checked_on": time.strftime("%Y-%m-%d"), "verdict": "unchecked",
             "sources": [{"name": "arxiv", "query": query, "hits": len(a)},
                         {"name": "openalex", "query": query, "hits": len(o)}],
             "conclusion": "FILL IN: novel / partial-prior-art / replication + your differentiator"}
    print("\nPaste into the registry row's novelty_check:\n")
    print(json.dumps(block, indent=2))


if __name__ == "__main__":
    main()
