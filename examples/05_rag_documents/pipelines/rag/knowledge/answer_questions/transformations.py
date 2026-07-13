"""Retrieve, then generate — and write down what was retrieved.

The order is the whole idea. Embed the question in the *same* vector space as the
chunks, find the nearest ones by cosine similarity, and hand only those to the
model as context. The model is told to answer from the context and to say so when
the context does not contain the answer.

Every answer is stored with the chunk ids that produced it. An answer you cannot
trace back to its sources is not a retrieval system, it is a rumour — and the
whole reason to build RAG rather than just ask the model is that you wanted the
sources.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Tuple

import numpy as np
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql import types as T
from ubunye.core.interfaces import Task

log = logging.getLogger(__name__)

# Both open-source, both served in-workspace.
EMBEDDING_ENDPOINT = os.environ.get("EMBEDDING_ENDPOINT", "databricks-bge-large-en")
# Llama 3.1 8B — open source, and it returns plain text. gpt-oss-20b returns
# structured *reasoning* objects instead of a string, which every naive
# `choices[0].message.content` in the world would mangle.
LLM_ENDPOINT = os.environ.get("LLM_ENDPOINT", "databricks-meta-llama-3-1-8b-instruct")

TOP_K = 4

# The questions to answer. In a real system these arrive from an application; here
# they are a fixed set so the run is reproducible and its answers are checkable.
# The last one is deliberately unanswerable from the corpus — a RAG system that
# will not say "I don't know" is a RAG system that will confidently make things up.
QUESTIONS = [
    "Why is idempotency important in a data pipeline?",
    "What does a lakehouse give you that a data lake does not?",
    "Should a model be registered if it fails its quality gate?",
    "What breaks when you move to serverless compute?",
    "What is the airspeed velocity of an unladen swallow?",
]

SYSTEM_PROMPT = (
    "You answer questions using ONLY the context provided. "
    "If the context does not contain the answer, say exactly: "
    "'The provided context does not answer this question.' "
    "Do not use outside knowledge. Be concise — three sentences at most."
)


class AnswerQuestions(Task):
    """chunk_embeddings + questions -> answers, with their sources."""

    def transform(self, sources: Dict[str, Any]) -> Dict[str, Any]:
        embeddings: DataFrame = sources["embeddings"]

        rows = embeddings.select("chunk_id", "title", "chunk_text", "embedding").collect()
        if not rows:
            raise RuntimeError(
                "No embeddings found. Run the embed_chunks task first — there is "
                "nothing to retrieve from."
            )

        matrix = _normalise(np.array([r["embedding"] for r in rows], dtype=np.float32))
        log.info("retrieving over %d chunks (%d dims)", matrix.shape[0], matrix.shape[1])

        question_vectors = _embed(QUESTIONS)
        answers = []

        for question, q_vec in zip(QUESTIONS, question_vectors):
            hits = self._retrieve(q_vec, matrix, rows)
            context = "\n\n".join(f"[{i + 1}] {text}" for i, (_, _, text, _) in enumerate(hits))
            answer = _generate(question, context)

            answers.append(
                (
                    question,
                    answer,
                    [chunk_id for chunk_id, _, _, _ in hits],
                    [title for _, title, _, _ in hits],
                    [float(score) for _, _, _, score in hits],
                    float(hits[0][3]),
                    LLM_ENDPOINT,
                    EMBEDDING_ENDPOINT,
                )
            )
            log.info("Q: %s -> top score %.3f", question, hits[0][3])

        schema = T.StructType(
            [
                T.StructField("question", T.StringType()),
                T.StructField("answer", T.StringType()),
                T.StructField("source_chunk_ids", T.ArrayType(T.StringType())),
                T.StructField("source_titles", T.ArrayType(T.StringType())),
                T.StructField("similarity_scores", T.ArrayType(T.FloatType())),
                T.StructField("top_similarity", T.FloatType()),
                T.StructField("llm_endpoint", T.StringType()),
                T.StructField("embedding_endpoint", T.StringType()),
            ]
        )

        spark = embeddings.sparkSession
        return {
            "rag_answers": spark.createDataFrame(answers, schema).withColumn(
                "answered_at", F.current_timestamp()
            )
        }

    def _retrieve(
        self, q_vec: np.ndarray, matrix: np.ndarray, rows: List[Any]
    ) -> List[Tuple[str, str, str, float]]:
        """Cosine similarity, top-k.

        Both sides are L2-normalised, so the dot product IS the cosine. Skipping
        the normalisation is the classic bug: you end up ranking by vector
        magnitude, which mostly means you retrieve the longest chunks.
        """
        scores = matrix @ (q_vec / np.linalg.norm(q_vec))
        top = np.argsort(scores)[::-1][:TOP_K]
        return [
            (rows[i]["chunk_id"], rows[i]["title"], rows[i]["chunk_text"], float(scores[i]))
            for i in top
        ]


def _normalise(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0  # a zero vector would otherwise produce NaNs
    return matrix / norms


def _embed(texts: List[str]) -> List[np.ndarray]:
    """The questions must be embedded by the SAME model as the chunks.

    Two models mean two vector spaces, and a dot product between them is a number
    with no meaning — it will still rank, and the ranking will be noise.
    """
    from serving import embed

    return [np.array(v, dtype=np.float32) for v in embed(EMBEDDING_ENDPOINT, texts)]


def _generate(question: str, context: str) -> str:
    """Ask the open-source LLM, giving it the retrieved context and nothing else."""
    from serving import chat

    prompt = "Context:\n" + context + "\n\nQuestion: " + question
    return chat(LLM_ENDPOINT, system=SYSTEM_PROMPT, user=prompt)
