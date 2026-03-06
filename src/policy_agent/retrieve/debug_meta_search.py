import json
from pathlib import Path

def main():
    meta_path = Path("data/index/chunks_meta.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    hits = []
    for m in meta:
        t = (m.get("text") or "")
        if "200.403" in t:
            hits.append(m)

    print("chunks_meta.json total:", len(meta))
    print("chunks_meta.json hits containing '200.403' in text:", len(hits))

    # print first 2 hits so we can inspect what the retriever actually sees
    for i, h in enumerate(hits[:2], start=1):
        print("\n--- HIT", i, "---")
        print("chunk_id:", h.get("chunk_id"))
        print("section field:", h.get("section"))
        print("text length:", len(h.get("text") or ""))
        print("text preview:", (h.get("text") or "")[:300])

if __name__ == "__main__":
    main()
