"""Evaluate retrieval and answer quality against a running DocQA server.

    python eval/run_eval.py --base-url http://localhost:8000

Reports, per retrieval mode (vector / keyword / hybrid):
  hit@k      - at least one retrieved chunk contains the expected answer keywords
  answer_ok  - the answer contains the expected keywords (answerable questions)
  refusal_ok - unanswerable questions are refused (no hallucinated answer)
It also prints the best-match distance for answerable vs unanswerable questions,
which tells you how to set MAX_VECTOR_DISTANCE for your embedding model.

Add your own documents to eval/docs/ and questions to eval/questions.jsonl;
aim for 30+ questions before quoting numbers on a resume.
"""
import argparse
import json
import sys
import time
import uuid
from pathlib import Path

import httpx

HERE = Path(__file__).parent


def post_ask(client: httpx.Client, payload: dict) -> dict:
    """POST /ask, waiting out rate limits (429) instead of failing."""
    while True:
        r = client.post("/ask", json=payload)
        if r.status_code == 429:
            time.sleep(int(r.headers.get("retry-after", "5")) + 1)
            continue
        r.raise_for_status()
        return r.json()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--questions", default=str(HERE / "questions.jsonl"))
    parser.add_argument("--docs", default=str(HERE / "docs"))
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    client = httpx.Client(base_url=args.base_url, timeout=120)
    email, password = f"eval-{uuid.uuid4().hex[:8]}@example.com", "eval-password-123"
    client.post("/auth/register", json={"email": email, "password": password}).raise_for_status()
    token = client.post("/auth/login", data={"username": email, "password": password}).json()["access_token"]
    client.headers["Authorization"] = f"Bearer {token}"

    ids = []
    for path in sorted(Path(args.docs).iterdir()):
        r = client.post("/documents", files={"file": (path.name, path.read_bytes())})
        r.raise_for_status()
        ids.append(r.json()["id"])
    deadline = time.time() + 300
    while time.time() < deadline:
        statuses = {client.get(f"/documents/{i}").json()["status"] for i in ids}
        if statuses <= {"ready", "failed"}:
            break
        time.sleep(1)
    if statuses != {"ready"}:
        print(f"Ingestion did not finish cleanly: {statuses}")
        return 1

    questions = [json.loads(line) for line in Path(args.questions).read_text().splitlines() if line.strip()]
    answerable = [q for q in questions if q["answerable"]]
    unanswerable = [q for q in questions if not q["answerable"]]
    print(f"{len(answerable)} answerable + {len(unanswerable)} unanswerable questions, top_k={args.top_k}\n")
    print(f"{'mode':<9} {'hit@k':>7} {'answer_ok':>10} {'refusal_ok':>11} {'avg_ms':>8}")

    distances_ok, distances_bad = [], []
    for mode in ("vector", "keyword", "hybrid"):
        hits = answers = refusals = 0
        latencies = []
        for q in questions:
            body = post_ask(
                client, {"question": q["question"], "mode": mode, "top_k": args.top_k, "no_cache": True}
            )
            latencies.append(body["latency_ms"])
            retrieved = body["retrieved"]
            best = next((c["distance"] for c in retrieved if c["distance"] is not None), None)
            if q["answerable"]:
                hits += any(all(k.lower() in c["content"].lower() for k in q["keywords"]) for c in retrieved)
                answers += (not body["refused"]) and all(k.lower() in body["answer"].lower() for k in q["keywords"])
                if mode == "vector" and best is not None:
                    distances_ok.append(best)
            else:
                refusals += body["refused"]
                if mode == "vector" and best is not None:
                    distances_bad.append(best)
        n, m = max(len(answerable), 1), max(len(unanswerable), 1)
        avg_ms = sum(latencies) / len(latencies)
        print(f"{mode:<9} {hits / n:>7.0%} {answers / n:>10.0%} {refusals / m:>11.0%} {avg_ms:>8.0f}")

    if distances_ok and distances_bad:
        print(
            f"\nBest-match cosine distance: answerable max={max(distances_ok):.3f}, "
            f"unanswerable min={min(distances_bad):.3f}"
        )
        if max(distances_ok) < min(distances_bad):
            print("-> a MAX_VECTOR_DISTANCE between these two values separates them cleanly.")
        else:
            print("-> the ranges overlap; use more/better questions or a stronger embedding model.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
