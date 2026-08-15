"""Benchmark loading.

Both suites come from the preprocessed HuggingFace datasets that kvpress's own
evaluation uses, so our uncompressed baseline is comparable to their published
numbers. The dataset id and the `data_dir` convention are theirs; the sampling
and the record-keeping are ours.

Note on RULER: kvpress's copy is tokenised with the Llama-3.1 tokenizer for
every model, where the original RULER paper tokenises per model. That makes
these numbers comparable *across compression ratios*, which is what we need, but
not directly against the RULER paper. Said plainly here so the report can say it
plainly too.
"""

from __future__ import annotations

from dataclasses import dataclass, field

DATASET_IDS = {
    "longbench": "Xnhyacinth/LongBench",
    "ruler": "simonjegou/ruler",
}

# Spanning single-doc QA, multi-doc QA and summarisation, per the spec.
DEFAULT_LONGBENCH_TASKS = [
    "narrativeqa",
    "qasper",
    "hotpotqa",
    "2wikimqa",
    "gov_report",
    "triviaqa",
]


@dataclass
class Sample:
    """One prompt to run. `answers` is a list because most tasks allow several."""

    context: str
    question: str
    answer_prefix: str
    answers: list[str]
    task: str
    extra: dict = field(default_factory=dict)

    def prompt(self) -> str:
        return f"{self.context}{self.question}{self.answer_prefix}"


def data_dir_for(suite: str, task: str, context_length: int | None) -> str:
    """kvpress keys LongBench by task name and RULER by context length."""
    if suite == "ruler":
        if context_length is None:
            raise ValueError("ruler needs a context length")
        return str(context_length)
    return task


def load_samples(
    suite: str,
    task: str,
    *,
    context_length: int | None = None,
    limit: int | None = None,
    split: str = "test",
) -> list[Sample]:
    """Load a task, deterministically truncated to `limit` samples.

    Truncation takes the first N rows rather than a random subset: every method
    and budget must see exactly the same prompts, or quality differences are
    partly sampling noise.
    """
    if suite not in DATASET_IDS:
        raise KeyError(f"unknown benchmark suite '{suite}'. Known: {sorted(DATASET_IDS)}")

    from datasets import load_dataset

    dataset = load_dataset(
        DATASET_IDS[suite], data_dir=data_dir_for(suite, task, context_length), split=split
    )
    frame = dataset.to_pandas()

    if suite == "ruler":
        # One RULER data_dir holds all 13 tasks at that context length.
        frame = frame[frame["task"] == task]
        if frame.empty:
            available = sorted(dataset.to_pandas()["task"].unique())
            raise KeyError(f"ruler task '{task}' not present at {context_length}; have {available}")

    if limit is not None:
        frame = frame.head(limit)

    return [_to_sample(row, suite, task) for _, row in frame.iterrows()]


def _to_sample(row, suite: str, task: str) -> Sample:
    answers = row["answer"] if suite == "ruler" else row["answers"]
    extra = {}
    for column in ("all_classes", "length", "task"):
        if column in row.index:
            extra[column] = row[column]
    return Sample(
        context=str(row["context"]),
        question=str(row.get("question", "") or ""),
        answer_prefix=str(row.get("answer_prefix", "") or ""),
        answers=[str(a) for a in _as_list(answers)],
        task=str(row.get("task", task)),
        extra=extra,
    )


def _as_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return list(value)
