import json

import openai
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from llama_index.core.schema import NodeWithScore


@dataclass
class DocumentScore:
    doc: NodeWithScore
    score: int
    score_reason: str
    similarity_score: float

@dataclass
class BatchProcessingConfig:
    """Configuration for batch processing of documents."""

    batch_size: int = 15
    max_workers: int = 7


def score_relevance_batch(contexts: list[str], query: str) -> list[tuple[int, str]]:
    """Score multiple contexts in a single LLM request."""
    # Note: Asking the LLM to describe the relevance of each
    # context appears to improve its accuracy.

    # Some principles at play here:
    #  - We ask the LLM to give the reason first, then the score. This gives the LLM
    #    "time to think" about the score and give a more relevant one
    #  - We return json so that it's easier to parse the results
    batch_prompt = f"""You are an expert research assistant, tasked with identifying if the given pieces of information (the context) is relevant to the researcher's query. For each of the contexts listed below:

    1. Explain the relevance of each context to the 'Query' given below in 10 words or less. If it doesn't relate, say so
    2. Give a score from 0 to 10 of how relevant it is to the 'Query', with 10 being the most relevant.

    Score on a scale of 0-10 where:
        - 10: Perfect match, directly answers the query
        - 7-9: Highly relevant, contains most key information
        - 4-6: Moderately relevant, contains some related information
        - 1-3: Minimally relevant, only tangentially related
        - 0: Completely irrelevant

    Respond in json format with the following structure, with a list entry for each context:

    Example output:

    [
        {{"reason": "brief explanation of relevance", "score": 7}},
    ]

    Query: {query}

    {contexts}

    Scores:"""

    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": batch_prompt}],
        )
        scores_str = response.choices[0].message.content.strip()
        print(f"Raw result: {scores_str}")
        scores_str = scores_str.replace("```json", "").replace("```", "") # remove json demarcators
        scores = json.loads(scores_str)
        score_vals = [(int(score["score"]), score["reason"]) for score in scores]
        print(f"[INFO] Batch scored {len(score_vals)} contexts")

    except Exception as e:
        print(f"[ERROR] Batch scoring failed: {e}")
        return [0] * len(contexts)
    else:
        print(f"Got the scores {score_vals}")
        return score_vals


def batch_score_documents(docs, question, config):
    """Score documents in parallel batches."""
    scored_docs = []
    with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
        futures = []
        for i in range(0, len(docs), config.batch_size):
            batch = docs[i:i + config.batch_size]
            print(f"Processing batch {i//config.batch_size + 1}")

            futures.append(executor.submit(score_relevance_batch, [d.text for d in batch], question))

        for future in as_completed(futures):
            try:
                scores = future.result()
                batch = futures.index(future) * config.batch_size
                for doc, score in zip(docs[batch:batch + config.batch_size], scores):
                    print(f"Relevance score: {score[0]} - {score[1]}")
                    print(f"Text: {doc.text[:200]}\n\n")
                    scored_docs.append(DocumentScore(doc=doc, score=score[0], score_reason=score[1], similarity_score=doc.score))
            except Exception as e:  # noqa: PERF203
                print(f"Error scoring document: {e}")

    return scored_docs