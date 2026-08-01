"""
run_ragas.py
============
Evaluation script: runs the test question set through three retrieval
strategies (keyword-only / semantic-only / hybrid+reranked) and produces
a Ragas comparison chart.

This is a LOCAL/OFFLINE tool, not part of the deployed Streamlit app --
its dependencies (ragas, datasets, matplotlib) live in
eval/requirements-eval.txt, kept separate from the app's requirements.txt
so the deployed app's dependency footprint stays small.

HOW TO USE:
1. Fill in "relevant_chunk_ids" for each question in eval/test_queries.json
   (Ragas' context precision/recall metrics need this ground truth).
2. Build a knowledge base once via the app (or `python app/indexing.py`).
3. From the project root:  pip install -r eval/requirements-eval.txt
                            python eval/run_ragas.py
4. Results land in eval/results/.

METRICS:
- context_precision: of the chunks retrieved, how many were relevant
- context_recall: of all relevant chunks that exist, how many were retrieved
- faithfulness: does the answer only state things supported by the retrieved
  chunks (a second, model-based hallucination check, complementing
  citation.py's rule-based one)
- answer_relevancy: does the generated answer address the question asked
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

sys.path.append(str(Path(__file__).resolve().parent.parent / "app"))
from config import TOP_K_FINAL          # noqa: E402
from generation import build_rag_chain, generate_answer  # noqa: E402
from indexing import KnowledgeBase      # noqa: E402
from reranker import build_reranking_retriever  # noqa: E402
from retrieval import build_hybrid_retriever    # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def load_test_queries() -> list[dict]:
    path = Path(__file__).resolve().parent / "test_queries.json"
    with open(path) as f:
        return json.load(f)


def build_retriever_for_mode(kb: KnowledgeBase, mode: str):
    """Builds the base retriever for one of the three comparison modes."""
    if mode == "keyword":
        return kb.bm25_retriever
    if mode == "semantic":
        return kb.vector_store.as_retriever(search_kwargs={"k": TOP_K_FINAL})
    if mode == "hybrid":
        return build_reranking_retriever(build_hybrid_retriever(kb))
    raise ValueError(f"Unknown mode: {mode}")


def run_pipeline_for_mode(kb: KnowledgeBase, questions: list[str], mode: str) -> dict:
    retriever = build_retriever_for_mode(kb, mode)
    chain = build_rag_chain(retriever)

    answers, contexts = [], []
    for question in questions:
        result = generate_answer(chain, question, chat_history=[])
        answers.append(result["answer"])
        contexts.append([c.page_content for c in result["context"]])
    return {"question": questions, "answer": answers, "contexts": contexts}


def main():
    test_data = load_test_queries()
    if not test_data:
        print("eval/test_queries.json is empty -- add some test questions first.")
        return

    questions = [item["question"] for item in test_data]
    ground_truths = [item.get("relevant_chunk_ids", []) for item in test_data]

    kb = KnowledgeBase()
    kb.load()

    all_scores = {}

    for mode in ["keyword", "semantic", "hybrid"]:
        print(f"\n=== Evaluating mode: {mode} ===")
        pipeline_output = run_pipeline_for_mode(kb, questions, mode)

        ragas_dataset = Dataset.from_dict({
            "question": pipeline_output["question"],
            "answer": pipeline_output["answer"],
            "contexts": pipeline_output["contexts"],
            "ground_truth": [" ".join(gt) if gt else "" for gt in ground_truths],
        })

        result = evaluate(
            ragas_dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        )
        all_scores[mode] = result

        with open(RESULTS_DIR / f"{mode}_scores.json", "w") as f:
            json.dump(dict(result), f, indent=2)

    plot_comparison_chart(all_scores)


def plot_comparison_chart(all_scores: dict):
    metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    modes = list(all_scores.keys())

    x = range(len(metrics))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, mode in enumerate(modes):
        values = [all_scores[mode][m] for m in metrics]
        positions = [xi + i * width for xi in x]
        ax.bar(positions, values, width, label=mode)

    ax.set_xticks([xi + width for xi in x])
    ax.set_xticklabels(metrics, rotation=15)
    ax.set_ylabel("Score")
    ax.set_title("RAG Evaluation: Keyword vs Semantic vs Hybrid+Reranked Retrieval")
    ax.legend(title="Retrieval mode")
    ax.set_ylim(0, 1)
    fig.tight_layout()

    output_path = RESULTS_DIR / "comparison_chart.png"
    fig.savefig(output_path, dpi=200)
    print(f"\nSaved comparison chart to {output_path}")


if __name__ == "__main__":
    main()
