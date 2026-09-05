"""Getting records off the box, which is what actually protects a long sweep."""

import subprocess

import pytest

from kvbench.results import ResultStore
from kvbench.sync import ResultSync


def git(cwd, *args):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path):
    """A working repo with a real (bare, local) remote to push to."""
    remote = tmp_path / "remote.git"
    subprocess.run(["git", "init", "--bare", str(remote)], check=True, capture_output=True)

    work = tmp_path / "work"
    work.mkdir()
    git(work, "init")
    git(work, "config", "user.email", "test@example.com")
    git(work, "config", "user.name", "test")
    git(work, "remote", "add", "origin", str(remote))
    (work / "README.md").write_text("seed\n")
    git(work, "add", ".")
    git(work, "commit", "-m", "seed")
    git(work, "push", "-u", "origin", "HEAD")
    return work


def test_a_record_is_pushed_to_the_remote(repo, record_factory):
    store = ResultStore(repo / "results")
    store.save(record_factory("run-a"))

    syncer = ResultSync(store.root)
    assert syncer.available()[0]
    assert syncer.push("run-a") is True
    assert syncer.pushed == 1
    assert syncer.failures == []

    # Read the remote itself, not a local ref: the point of the exercise is that
    # the record left the machine.
    log = subprocess.run(
        ["git", "--git-dir", str(repo.parent / "remote.git"), "log", "--oneline", "-1"],
        capture_output=True,
        text=True,
    )
    assert "run-a" in log.stdout


def test_a_relative_results_path_still_pushes(repo, record_factory, monkeypatch):
    # How the CLI actually calls it: cwd is the repo root and the config names
    # "results". The sync runs git from inside that directory, so a relative
    # path handed straight to git add never matched anything.
    monkeypatch.chdir(repo)
    store = ResultStore("results")
    store.save(record_factory("run-rel"))

    syncer = ResultSync("results")
    assert syncer.push("run-rel") is True, syncer.failures
    assert syncer.failures == []


def test_pushing_with_nothing_new_is_a_no_op(repo, record_factory):
    store = ResultStore(repo / "results")
    store.save(record_factory("run-a"))
    syncer = ResultSync(store.root)
    syncer.push("run-a")

    assert syncer.push("run-a") is True
    assert syncer.pushed == 1  # not counted twice


def test_a_missing_remote_is_caught_before_the_sweep_starts(tmp_path):
    work = tmp_path / "work"
    (work / "results").mkdir(parents=True)
    git_dir = work
    subprocess.run(["git", "init"], cwd=git_dir, check=True, capture_output=True)

    ok, detail = ResultSync(work / "results").available()
    assert not ok
    assert "remote" in detail


def test_a_non_repository_is_caught_before_the_sweep_starts(tmp_path):
    results = tmp_path / "loose" / "results"
    results.mkdir(parents=True)
    ok, detail = ResultSync(results).available()
    assert not ok
    assert "git repository" in detail


def test_a_failed_push_is_recorded_but_never_raises(repo, record_factory, monkeypatch):
    store = ResultStore(repo / "results")
    store.save(record_factory("run-a"))
    syncer = ResultSync(store.root, remote="nonexistent")

    # A dropped network on a rented box is routine; losing the sweep over it
    # would not be.
    assert syncer.push("run-a") is False
    assert syncer.failures
    assert "sync failures" in syncer.summary()


def test_the_summary_reports_success_honestly(repo, record_factory):
    store = ResultStore(repo / "results")
    store.save(record_factory("run-a"))
    syncer = ResultSync(store.root)
    syncer.push("run-a")
    assert "synced 1 records" in syncer.summary()
