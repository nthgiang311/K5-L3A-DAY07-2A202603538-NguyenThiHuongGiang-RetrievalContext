import argparse
import json
import os
import statistics
import time
from pathlib import Path

import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer

from baseline import validate, hit, pack_context


DAY6_ENV = Path(
    "/mnt/d/K5-L3A-DAY06-NguyenThiHuonGiang-2A202603538-DataIngestion/starter/.env"
)
DOCUMENT_ID = "openstax_calculus_v1"
STRATEGY = "A"


class DenseIndex:
    def __init__(self, chunks):
        load_dotenv(DAY6_ENV)

        database_url = os.getenv("DATABASE_URL")
        model_name = os.getenv(
            "EMBEDDING_MODEL",
            "sentence-transformers/all-MiniLM-L6-v2"
        )

        if not database_url:
            raise RuntimeError("DATABASE_URL not found in Day 06 .env")

        self.model = SentenceTransformer(model_name)
        self.conn = psycopg.connect(database_url)
        register_vector(self.conn)

        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM chunks WHERE strategy=%s AND document_id=%s",
                (STRATEGY, DOCUMENT_ID)
            )
            db_count = cur.fetchone()[0]

        expected_count = len(chunks)
        if db_count != expected_count:
            self.conn.close()
            raise RuntimeError(
                f"Chunk count mismatch: database={db_count}, JSON={expected_count}"
            )

        print(
            f"Dense ready: model={model_name}, "
            f"dimensions={self.model.get_embedding_dimension()}, "
            f"chunks={db_count}"
        )

    def search(self, question, top_k=3):
        query_vector = self.model.encode(
            question,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        ).tolist()

        sql = """
            SELECT chunk_id, document_id, source, page, text, metadata,
       		   1 - (embedding <=> %s::vector) AS score
	    FROM chunks
	    WHERE strategy = %s AND document_id = %s
	    ORDER BY embedding <=> %s::vector
	    LIMIT %s
        """

        with self.conn.cursor() as cur:
            cur.execute(
                sql,
                (query_vector, STRATEGY, DOCUMENT_ID, query_vector, top_k)
            )
            rows = cur.fetchall()

        results = []
        for row in rows:
            chunk_id, document_id, source, page, text, metadata, score = row
            results.append({
                "id": chunk_id,
                "source_id": document_id,
                "source": source,
                "page": page,
                "text": text,
                "metadata": metadata,
                "score": float(score),
                "strategy": "dense"
            })

        return results

    def close(self):
        self.conn.close()


def evaluate(index, questions, budget=1800):
    traces = []

    for q in questions:
        start = time.perf_counter()
        results = index.search(q["question"], top_k=3)
        context, ids = pack_context(results, budget)
        elapsed_ms = (time.perf_counter() - start) * 1000

        marked = [
            {**c, "hit": hit(c, q["gold"])}
            for c in results
        ]

        traces.append({
            "query_id": q["id"],
            "split": q["split"],
            "question": q["question"],
            "strategy": "dense",
            "top_k": 3,
            "budget_chars": budget,
            "gold": q["gold"],
            "results": marked,
            "context": context,
            "context_ids": ids,
            "context_chars": len(context),
            "elapsed_ms": elapsed_ms,
            "hit1": bool(marked and marked[0]["hit"]),
            "hit3": any(c["hit"] for c in marked),
            "context_hit": any(
                c["hit"] and c["id"] in ids for c in marked
            )
        })

    return traces


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", required=True, type=Path)
    parser.add_argument("--questions", required=True, type=Path)
    parser.add_argument("--split", required=True, choices=("dev", "test"))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    chunks = json.loads(args.chunks.read_text(encoding="utf-8"))
    questions = json.loads(args.questions.read_text(encoding="utf-8"))

    selected = [
	    q for q in questions
	    if q.get("split") == args.split
	    and q.get("answerable", True)
	]

    validate(chunks, selected)

    started = time.perf_counter()
    index = DenseIndex(chunks)
    index_seconds = time.perf_counter() - started

    try:
        traces = evaluate(index, selected)
    finally:
        index.close()

    summary = {
        "strategy": "dense",
        "split": args.split,
        "n": len(traces),
        "hit1": sum(t["hit1"] for t in traces),
        "hit3": sum(t["hit3"] for t in traces),
        "context_hit": sum(t["context_hit"] for t in traces),
        "avg_context_chars": statistics.mean(
            t["context_chars"] for t in traces
        ),
        "median_ms": statistics.median(
            t["elapsed_ms"] for t in traces
        ),
        "index_seconds": index_seconds,
        "service_cost_usd": 0
    }

    args.output.mkdir(parents=True, exist_ok=True)

    (args.output / "traces.json").write_text(
        json.dumps(traces, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8"
    )

    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8"
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
