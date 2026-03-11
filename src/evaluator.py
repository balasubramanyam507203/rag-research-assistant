import json
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd

from src.config import BASE_DIR
from src.rag_pipeline import generate_rag_answer


EVAL_FILE = BASE_DIR / "evaluation" / "eval_questions.json"
RESULTS_FILE = BASE_DIR / "evaluation" / "evaluation_results.csv"


def load_evaluation_questions() -> List[Dict[str, str]]:
    """
    Load evaluation questions from JSON file.
    """
    with open(EVAL_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


def evaluate_rag_system() -> pd.DataFrame:
    """
    Run evaluation questions through the RAG pipeline and save results.
    """
    eval_questions = load_evaluation_questions()
    results: List[Dict[str, Any]] = []

    for item in eval_questions:
        question = item["question"]
        expected_source = item["expected_source"]

        result = generate_rag_answer(
            query=question,
            top_k_dense=5,
            top_k_bm25=5,
            top_k_final=3,
            num_variations=3,
        )

        retrieved_sources = [chunk["source"] for chunk in result["retrieved_chunks"]]
        source_hit = expected_source in retrieved_sources

        results.append(
            {
                "question": question,
                "expected_source": expected_source,
                "retrieved_sources": " | ".join(retrieved_sources),
                "source_hit": source_hit,
                "answer": result["answer"],
            }
        )

    df = pd.DataFrame(results)
    df.to_csv(RESULTS_FILE, index=False)
    return df