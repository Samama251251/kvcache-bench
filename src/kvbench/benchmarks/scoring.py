"""Quality scoring, using each benchmark's own automatic metric.

No LLM-as-a-judge anywhere: LongBench tasks are scored with their native F1 /
ROUGE-L / classification / retrieval metrics, RULER with substring match. The
scoring functions themselves are kvpress's, vendored unmodified, so our
uncompressed baseline lands where their published baseline lands.
"""

from __future__ import annotations

import pandas as pd

from .loader import Sample
from .vendored import longbench_metrics, ruler_metrics

# Which metric each LongBench task is actually scored by, for labelling plots.
# Mirrors the dataset2metric table in the vendored scorer.
LONGBENCH_METRIC_NAMES = {
    "narrativeqa": "qa_f1",
    "qasper": "qa_f1",
    "multifieldqa_en": "qa_f1",
    "hotpotqa": "qa_f1",
    "2wikimqa": "qa_f1",
    "musique": "qa_f1",
    "triviaqa": "qa_f1",
    "gov_report": "rouge_l",
    "qmsum": "rouge_l",
    "multi_news": "rouge_l",
    "samsum": "rouge_l",
    "trec": "classification",
    "lsht": "classification",
    "passage_retrieval_en": "retrieval",
    "passage_count": "count",
    "lcc": "code_sim",
    "repobench-p": "code_sim",
}


def score(
    suite: str, task: str, samples: list[Sample], predictions: list[str]
) -> tuple[str, float | None, dict]:
    """Return (primary metric name, primary score, all scores)."""
    if len(samples) != len(predictions):
        raise ValueError("one prediction per sample is required for scoring")
    if not samples:
        return (metric_name(suite, task), None, {})

    frame = _frame(suite, task, samples, predictions)
    if suite == "ruler":
        scores = ruler_metrics.calculate_metrics(frame)
        primary = scores.get(task, {}).get("string_match")
        return ("string_match", _as_float(primary), scores)

    value = longbench_metrics.calculate_metrics(frame)
    name = metric_name(suite, task)
    return (name, _as_float(value), {name: _as_float(value)})


def metric_name(suite: str, task: str) -> str:
    if suite == "ruler":
        return "string_match"
    return LONGBENCH_METRIC_NAMES.get(task, "longbench_score")


def _frame(suite: str, task: str, samples: list[Sample], predictions: list[str]) -> pd.DataFrame:
    rows = []
    for sample, prediction in zip(samples, predictions, strict=True):
        row = {
            "task": sample.task or task,
            "predicted_answer": prediction,
            # The two scorers disagree on the column name for ground truth.
            "answer": sample.answers,
            "answers": sample.answers,
            "all_classes": sample.extra.get("all_classes"),
            "length": sample.extra.get("length", 0),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def _as_float(value) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
