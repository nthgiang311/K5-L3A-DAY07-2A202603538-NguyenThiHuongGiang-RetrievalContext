
import re
import unicodedata
from rank_bm25 import BM25Okapi


def tokenize(text):
    return re.findall(
        r'\w+',
        unicodedata.normalize('NFC', text).casefold()
    )


class BM25Index:
    def __init__(self, chunks):
        self.chunks = chunks
        self.tokenized_chunks = [
            tokenize(chunk['text']) for chunk in chunks
        ]
        self.index = BM25Okapi(self.tokenized_chunks)

    def search(self, question, top_k=3):
        query = tokenize(question)
        scores = self.index.get_scores(query)

        ranked = sorted(
            enumerate(scores),
            key=lambda item: (-item[1], self.chunks[item[0]]['id'])
        )

        results = []
        for idx, score in ranked:
            if score <= 0:
                continue

            results.append({
                **self.chunks[idx],
                'score': float(score),
                'strategy': 'bm25'
            })

            if len(results) >= top_k:
                break

        return results
