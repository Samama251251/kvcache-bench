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

import glob
import os
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
    # The benchmark's own generation budget for this example. LongBench and
    # RULER both ship it per row, and using it rather than one flat number is
    # what makes our scores comparable to published ones -- gov_report wants 512
    # tokens where hotpotqa wants 32.
    max_new_tokens: int | None = None
    extra: dict = field(default_factory=dict)

    def prompt(self, tokenizer=None) -> str:
        """The text the model sees, formatted the way kvpress formats it.

        With a chat tokenizer the context and question sit inside the user
        turn, then the assistant header, then the answer prefix. Without one it
        is plain concatenation. Instruct models without their template ramble,
        and the score they get is a fact about the prompt, not the method.
        """
        head, tail = _template_parts(tokenizer)
        return f"{head}{self.context}{self.question}{tail}{self.answer_prefix}"


def uses_chat_template(tokenizer) -> bool:
    return tokenizer is not None and getattr(tokenizer, "chat_template", None) is not None


_TEMPLATE_CACHE: dict[int, tuple[str, str]] = {}


def _template_parts(tokenizer) -> tuple[str, str]:
    """(prefix before the context, suffix after the question) for a tokenizer.

    Mirrors kvpress's pipeline: render one user turn containing a marker, split
    on the marker, and the two halves are what goes around context + question.
    For Llama-3.1 the suffix is the eot token plus the assistant header. The
    prefix already carries the BOS token, so callers must not add another.
    """
    if not uses_chat_template(tokenizer):
        return "", ""
    key = id(tokenizer)
    if key not in _TEMPLATE_CACHE:
        marker = "#" * 64
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": marker}], add_generation_prompt=True, tokenize=False
        )
        head, tail = rendered.split(marker)
        _TEMPLATE_CACHE[key] = (head, tail)
    return _TEMPLATE_CACHE[key]


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

    files = dataset_files(suite, data_dir_for(suite, task, context_length))
    dataset = load_dataset("parquet", data_files=files, split="train")
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


def dataset_files(suite: str, data_dir: str) -> list[str]:
    """Local parquet files for one task, fetched once and then read from disk.

    `load_dataset(repo, data_dir=...)` asks the Hub on every call. Unauthenticated
    the Hub rate-limits after a few dozen, and the datasets library then falls
    back to its cache under a config name that does not match, so cell 40 of a
    sweep fails on data cell 1 loaded fine. Snapshotting the task's directory
    makes the Hub a one-time dependency; after that it is a glob.
    """
    from huggingface_hub import snapshot_download

    repo = DATASET_IDS[suite]
    patterns = [f"{data_dir}/*"]
    try:
        local = snapshot_download(repo, repo_type="dataset", allow_patterns=patterns)
    except Exception:  # offline, rate-limited, or the Hub is down: use what we have
        local = snapshot_download(
            repo, repo_type="dataset", allow_patterns=patterns, local_files_only=True
        )
    files = sorted(glob.glob(os.path.join(local, data_dir, "*.parquet")))
    if not files:
        raise FileNotFoundError(f"{repo} has no parquet files under '{data_dir}/'")
    return files


def _to_sample(row, suite: str, task: str) -> Sample:
    answers = row["answer"] if suite == "ruler" else row["answers"]
    extra = {}
    for column in ("all_classes", "length", "task"):
        if column in row.index:
            extra[column] = row[column]
    declared = row.get("max_new_tokens")
    return Sample(
        context=str(row["context"]),
        question=str(row.get("question", "") or ""),
        answer_prefix=str(row.get("answer_prefix", "") or ""),
        answers=[str(a) for a in _as_list(answers)],
        task=str(row.get("task", task)),
        max_new_tokens=int(declared) if declared is not None else None,
        extra=extra,
    )


def _as_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return list(value)
