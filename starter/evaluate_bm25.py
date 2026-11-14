
import argparse
import json
import statistics
import time
from pathlib import Path

from baseline import validate, hit, pack_context
from retrieval import BM25Index


def evaluate(index, questions, budget=1800):
    traces = []

    for q in questions:
        start = time.perf_counter()
        results = index.search(q['question'], top_k=3)
        context, ids = pack_context(results, budget)
        elapsed_ms = (time.perf_counter() - start) * 1000

        marked = [
            {**c, 'hit': hit(c, q['gold'])}
            for c in results
        ]

        traces.append({
            'query_id': q['id'],
            'split': q['split'],
            'question': q['question'],
            'strategy': 'bm25',
            'top_k': 3,
            'budget_chars': budget,
            'gold': q['gold'],
            'results': marked,
            'context': context,
            'context_ids': ids,
            'context_chars': len(context),
            'elapsed_ms': elapsed_ms,
            'hit1': bool(marked and marked[0]['hit']),
            'hit3': any(c['hit'] for c in marked),
            'context_hit': any(
                c['hit'] and c['id'] in ids for c in marked
            )
        })

    return traces


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--chunks', required=True, type=Path)
    parser.add_argument('--questions', required=True, type=Path)
    parser.add_argument('--split', required=True, choices=('dev', 'test'))
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()

    chunks = json.loads(args.chunks.read_text(encoding='utf-8'))
    questions = json.loads(args.questions.read_text(encoding='utf-8'))

    selected = [
        q for q in questions if q.get('split') == args.split
    ]

    validate(chunks, selected)

    started = time.perf_counter()
    index = BM25Index(chunks)
    index_seconds = time.perf_counter() - started

    traces = evaluate(index, selected)

    summary = {
        'strategy': 'bm25',
        'split': args.split,
        'n': len(traces),
        'hit1': sum(t['hit1'] for t in traces),
        'hit3': sum(t['hit3'] for t in traces),
        'context_hit': sum(t['context_hit'] for t in traces),
        'avg_context_chars': statistics.mean(
            t['context_chars'] for t in traces
        ),
        'median_ms': statistics.median(
            t['elapsed_ms'] for t in traces
        ),
        'index_seconds': index_seconds,
        'service_cost_usd': 0
    }

    args.output.mkdir(parents=True, exist_ok=True)

    (args.output / 'traces.json').write_text(
        json.dumps(traces, ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8'
    )

    (args.output / 'summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8'
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
