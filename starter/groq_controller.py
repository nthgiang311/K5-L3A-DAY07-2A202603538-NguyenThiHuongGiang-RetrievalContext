
import os
import json
import urllib.request
import urllib.error
from pathlib import Path
from dotenv import load_dotenv


DAY7_ENV = Path(
    "/mnt/d/K5-L3A-DAY07-2A202603538-NguyenThiHuongGiang-RetrievalContext/.env"
)

load_dotenv(DAY7_ENV, override=True)


class GroqController:
    def __init__(self, model="openai/gpt-oss-20b"):
        self.api_key = os.getenv("GROQ_API_KEY")

        if not self.api_key:
            raise RuntimeError(
                "GROQ_API_KEY not found in Day 07 .env"
            )

        self.model = model
        self.url = "https://api.groq.com/openai/v1/chat/completions"

    def decide(self, question, history):
        context = ""
        context_ids = []
        previous_queries = []

        if history:
            context = history[-1].get("context", "")
            context_ids = history[-1].get("context_ids", [])
            previous_queries = history[-1].get("queries", [])

        prompt = f"""
You are a retrieval-augmented question answering agent.

Answer the user's question using ONLY the supplied context.

USER QUESTION:
{question}

PREVIOUS SEARCH QUERIES:
{json.dumps(previous_queries)}

AVAILABLE CONTEXT:
{context}

AVAILABLE CHUNK IDS:
{json.dumps(context_ids)}

RULES:
1. If there is no context, choose search.
2. If the context does not directly answer the question, choose search with a more specific query.
3. If a context chunk contains readable evidence that directly answers the question, choose answer.
4. Do not abstain because unrelated text is truncated or has OCR errors.
5. Never invent facts or use outside knowledge.
6. Every factual claim must be supported by the context.
7. Include exact chunk IDs in square brackets in the answer.
8. The citations list must contain only exact IDs from AVAILABLE CHUNK IDS.
9. Only abstain if sufficient evidence cannot be found after searching.
10. Return exactly one valid JSON object without markdown fences.

For search:
{{"action": "search", "query": "specific search query"}}

For answer:
{{"action": "answer", "answer": "answer with [chunk_id] citations", "citations": ["chunk_id"]}}
"""

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0,
            "max_completion_tokens": 512,
            "reasoning_effort": "low",
            "include_reasoning": False,
            "response_format": {
                "type": "json_object"
            }
        }

        request = urllib.request.Request(
            self.url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                "User-Agent": "day7-lab/1.0"
            },
            method="POST"
        )

        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                result = json.loads(
                    response.read().decode("utf-8")
                )

            content = result["choices"][0]["message"].get("content")

            if not content:
                raise RuntimeError(
                    "Groq returned empty message content"
                )

            action = json.loads(content)

            if action.get("action") == "search":
                query = action.get("query")

                if not isinstance(query, str) or not query.strip():
                    raise ValueError(
                        "Groq returned an invalid search query"
                    )

            elif action.get("action") == "answer":
                answer = action.get("answer")
                citations = action.get("citations")

                if not isinstance(answer, str):
                    raise ValueError(
                        "Groq returned an invalid answer"
                    )

                if not isinstance(citations, list):
                    raise ValueError(
                        "Groq returned invalid citations"
                    )

                if any(not isinstance(cid, str) for cid in citations):
                    raise ValueError(
                        "Citation IDs must be strings"
                    )

                # Ensure citations refer to chunks in the current context.
                valid_citations = [
                    cid for cid in citations
                    if cid in context_ids
                ]

                if not valid_citations:
                    raise ValueError(
                        "No valid citation found in the current context"
                    )

                # Add citation IDs to the answer if the model omitted them.
                for cid in valid_citations:
                    if f"[{cid}]" not in answer:
                        answer += f" [{cid}]"

                # Update the action returned to the agent loop.
                action["answer"] = answer
                action["citations"] = valid_citations

            else:
                raise ValueError(
                    "Groq returned an unknown action"
                )

            print(
                "GROQ_ACTION:",
                json.dumps(action, ensure_ascii=False)
            )

            return action

        except urllib.error.HTTPError as error:
            detail = error.read().decode(
                "utf-8",
                errors="replace"
            )

            raise RuntimeError(
                f"Groq API HTTP {error.code}: {detail}"
            ) from error

        except urllib.error.URLError as error:
            raise RuntimeError(
                f"Groq connection error: {error}"
            ) from error

        except json.JSONDecodeError as error:
            raise RuntimeError(
                f"Groq returned invalid JSON: {error}"
            ) from error
