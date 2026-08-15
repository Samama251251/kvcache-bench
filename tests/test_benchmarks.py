import pytest

from kvbench.benchmarks.loader import Sample, data_dir_for
from kvbench.benchmarks.scoring import metric_name, score


def sample(answers, task="qasper", **extra):
    return Sample(
        context="ctx ",
        question="q?",
        answer_prefix="\nAnswer:",
        answers=answers,
        task=task,
        extra=extra,
    )


def test_longbench_is_keyed_by_task_and_ruler_by_context_length():
    assert data_dir_for("longbench", "qasper", None) == "qasper"
    assert data_dir_for("ruler", "niah_single_1", 4096) == "4096"


def test_ruler_without_a_context_length_is_an_error():
    with pytest.raises(ValueError, match="context length"):
        data_dir_for("ruler", "niah_single_1", None)


def test_prompt_is_context_question_and_prefix_in_order():
    assert sample(["x"]).prompt() == "ctx q?\nAnswer:"


def test_a_perfect_longbench_answer_scores_full_marks():
    name, primary, scores = score("longbench", "qasper", [sample(["blue whale"])], ["blue whale"])
    assert name == "qa_f1"
    assert primary == pytest.approx(100.0)
    assert scores == {"qa_f1": pytest.approx(100.0)}


def test_a_wrong_longbench_answer_scores_zero():
    _, primary, _ = score("longbench", "qasper", [sample(["blue whale"])], ["a rock"])
    assert primary == pytest.approx(0.0)


def test_longbench_credits_the_best_of_several_references():
    samples = [sample(["a rock", "blue whale"])]
    _, primary, _ = score("longbench", "qasper", samples, ["blue whale"])
    assert primary == pytest.approx(100.0)


def test_summarisation_tasks_are_scored_by_rouge_not_f1():
    assert metric_name("longbench", "gov_report") == "rouge_l"
    name, primary, _ = score(
        "longbench",
        "gov_report",
        [sample(["the report describes federal spending"], task="gov_report")],
        ["the report describes federal spending"],
    )
    assert name == "rouge_l"
    assert primary == pytest.approx(100.0)


def test_ruler_scores_by_substring_match():
    samples = [
        sample(["7f3a"], task="niah_single_1"),
        sample(["9c1b"], task="niah_single_1"),
    ]
    name, primary, scores = score("ruler", "niah_single_1", samples, ["the magic number is 7f3a", "no idea"])
    assert name == "string_match"
    assert primary == pytest.approx(50.0)
    assert scores["niah_single_1"]["string_match"] == pytest.approx(50.0)


def test_ruler_matching_ignores_case():
    samples = [sample(["7F3A"], task="niah_single_1")]
    _, primary, _ = score("ruler", "niah_single_1", samples, ["...7f3a..."])
    assert primary == pytest.approx(100.0)


def test_scoring_nothing_yields_a_null_score_not_a_crash():
    name, primary, scores = score("longbench", "qasper", [], [])
    assert name == "qa_f1"
    assert primary is None
    assert scores == {}


def test_a_prediction_count_mismatch_is_rejected():
    with pytest.raises(ValueError, match="one prediction per sample"):
        score("longbench", "qasper", [sample(["x"]), sample(["y"])], ["only one"])


def test_a_benchmarks_own_generation_budget_is_carried_on_the_sample():
    # LongBench ships a per-task budget: gov_report wants 512 tokens where
    # hotpotqa wants 32. Using it is what keeps scores comparable to published
    # numbers.
    assert Sample("c", "q", "a", ["x"], "qasper").max_new_tokens is None
    assert Sample("c", "q", "a", ["x"], "gov_report", max_new_tokens=512).max_new_tokens == 512
