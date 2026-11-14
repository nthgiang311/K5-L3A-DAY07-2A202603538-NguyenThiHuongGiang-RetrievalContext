
import sys
import json
import argparse
import time
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
DAY7_STARTER = Path(__file__).resolve().parent

DAY6_STARTER = Path(
    "/mnt/d/K5-L3A-DAY06-NguyenThiHuonGiang-2A202603538-DataIngestion/starter"
)
DAY6_ENV = DAY6_STARTER / ".env"

sys.path.insert(0, str(DAY6_STARTER))
sys.path.insert(0, str(DAY7_STARTER))

load_dotenv(DAY6_ENV, override=False)

from src.embeddings import MyEmbedder
from src.store import PgVectorStore
from groq_controller import GroqController
from agentic import run_agent


class DenseRetriever:
    def __init__(self):
        import os

        database_url = os.getenv("DATABASE_URL")
        model_name = os.getenv("EMBEDDING_MODEL")

        if not database_url or not model_name:
            raise RuntimeError(
                "Missing DATABASE_URL or EMBEDDING_MODEL in Day 06 .env"
            )

        self.embedder = MyEmbedder(model_name)
        self.store = PgVectorStore(
            database_url,
            self.embedder.dimensions
        )

    def search(self, query, *, filters=None, top_k=3):
        vector = self.embedder.embed_query(query)

        rows = self.store.search(
            strategy="A",
            query_vector=vector,
            top_k=top_k,
            filters=filters
        )

        return [
            {
                "id": row["chunk_id"],
                "source_id": row["document_id"],
                "source": row["source"],
                "page": row["page"],
                "text": row["text"],
                "metadata": row["metadata"],
                "score": row["score"]
            }
            for row in rows
        ]


def run_one(question, controller, retriever, args):
    return run_agent(
        question,
        controller,
        retriever,
        max_searches=args.max_searches,
        top_k=args.top_k,
        budget=args.budget
    )


def main():
    parser = argparse.ArgumentParser()

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--question")
    mode.add_argument("--questions", type=Path)

    parser.add_argument("--split", choices=("dev", "test"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-searches", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--budget", type=int, default=1800)
    parser.add_argument("--delay-seconds", type=float, default=10)
    parser.add_argument("--retry-errors", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)

    args = parser.parse_args()

    if args.questions and not args.split:
        parser.error("--questions requires --split dev or --split test")

    if args.retry_errors and not args.questions:
        parser.error("--retry-errors requires --questions")

    if args.delay_seconds < 0:
        parser.error("--delay-seconds cannot be negative")

    print("Loading Groq Controller...")
    controller = GroqController()

    print("Loading Day 06 Dense Retriever...")
    retriever = DenseRetriever()

    if args.question:
        print("Running single-question agent...")

        try:
            result = run_one(
                args.question, controller, retriever, args
            )
            output = {"question": args.question, **result}

        except Exception as error:
            output = {
                "question": args.question,
                "status": "error",
                "error": str(error),
                "events": [],
                "citations": []
            }

        rendered = json.dumps(
            output, ensure_ascii=False, indent=2
        )
        print(rendered)

    else:
        all_cases = json.loads(
            args.questions.read_text(encoding="utf-8")
        )

        cases = [
            case for case in all_cases
            if case.get("split") == args.split
        ]

        if args.limit is not None:
            cases = cases[:args.limit]

        previous_by_id = {}

        if args.retry_errors:
            previous = json.loads(
                args.retry_errors.read_text(encoding="utf-8")
            )

            previous_by_id = {
                trace["query_id"]: trace
                for trace in previous
            }

            cases = [
                case for case in cases
                if previous_by_id.get(
                    case["id"], {}
                ).get("status") == "error"
            ]

            print(
                f"Retry mode: found {len(cases)} failed questions."
            )

        elif not cases:
            raise ValueError(
                f"No questions found for split: {args.split}"
            )

        traces = []

        if cases:
            print(
                f"Running batch: {len(cases)} questions "
                f"(split={args.split})"
            )

            for index, case in enumerate(cases, start=1):
                case_id = case["id"]
                question = case["question"]

                print(
                    f"\n[{index}/{len(cases)}] "
                    f"{case_id}: {question}"
                )

                if index > 1 and args.delay_seconds > 0:
                    print(
                        f"Waiting {args.delay_seconds} seconds..."
                    )
                    time.sleep(args.delay_seconds)

                try:
                    result = run_one(
                        question, controller, retriever, args
                    )

                    trace = {
                        "query_id": case_id,
                        "status": result["status"],
                        "events": result.get("events", []),
                        "citations": result.get("citations", []),
                        "answer": result.get("answer", "")
                    }

                except Exception as error:
                    trace = {
                        "query_id": case_id,
                        "status": "error",
                        "events": [],
                        "citations": [],
                        "error": str(error)
                    }

                traces.append(trace)
                print("Status:", trace["status"])

                if trace["status"] == "error":
                    print("Error:", trace.get("error", "Unknown error"))

        # Merge new results with previous results.
        for trace in traces:
            previous_by_id[trace["query_id"]] = trace

        # Preserve benchmark order.
        output = [
            previous_by_id[case["id"]]
            for case in all_cases
            if case.get("split") == args.split
            and case["id"] in previous_by_id
        ]

        print("\nBATCH SUMMARY")
        print("Total:", len(output))
        print(
            "Answered:",
            sum(t["status"] == "answered" for t in output)
        )
        print(
            "Abstained:",
            sum(t["status"] == "abstain" for t in output)
        )
        print(
            "Errors:",
            sum(t["status"] == "error" for t in output)
        )

        rendered = json.dumps(
            output, ensure_ascii=False, indent=2
        )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
        print(f"\nSaved to: {args.output}")


if __name__ == "__main__":
    main()
