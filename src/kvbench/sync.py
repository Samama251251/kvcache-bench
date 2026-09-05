"""Getting records off the box while the sweep is still running.

Atomic per-run writes protect against the *process* dying. They do nothing about
the *machine* dying, and on a rented, preemptible box that is the likelier of
the two: a destroyed instance takes 200 records with it however carefully each
one was written.

So every record is pushed to a git remote as it lands. It costs a second or two
per run against a 15-hour sweep, and it caps the worst case at losing the single
run in flight.
"""

from __future__ import annotations

import subprocess
from pathlib import Path


class ResultSync:
    """Commits and pushes the results directory after each run.

    Failures never interrupt a sweep. A dropped network on a rented box is
    routine; losing 15 hours of GPU time because a push failed is not. Failures
    are counted and reported at the end so a silent no-op cannot masquerade as a
    working backup.
    """

    def __init__(self, root: str | Path, remote: str = "origin", branch: str | None = None) -> None:
        self.root = Path(root)
        self.remote = remote
        self.branch = branch
        self.pushed = 0
        self.failures: list[str] = []

    def available(self) -> tuple[bool, str]:
        """Whether pushing can work at all, checked before the sweep starts."""
        if not self._git("rev-parse", "--git-dir").ok:
            return False, "not a git repository"
        if not self._git("remote", "get-url", self.remote).ok:
            return False, f"no git remote named '{self.remote}'"
        return True, f"pushing each record to {self.remote}"

    def push(self, label: str = "") -> bool:
        # Absolute, because git runs from inside the results directory and a
        # relative "results" resolves to results/results, which does not exist.
        # The CLI hands over a relative path from the config, so this is the
        # normal case, not an edge.
        add = self._git("add", "--", str(self.root.resolve()))
        if not add.ok:
            return self._fail(f"git add failed: {add.err}")

        # `git diff --cached --quiet` exits 0 when nothing is staged.
        if self._git("diff", "--cached", "--quiet").ok:
            return True

        message = f"Add results for {label}" if label else "Add sweep results"
        commit = self._git("commit", "-m", message)
        if not commit.ok:
            return self._fail(f"git commit failed: {commit.err}")

        args = ["push", self.remote]
        if self.branch:
            args.append(f"HEAD:{self.branch}")
        push = self._git(*args)
        if not push.ok:
            # The commit is still local, so the next successful push carries it.
            return self._fail(f"git push failed: {push.err}")

        self.pushed += 1
        return True

    def summary(self) -> str:
        if not self.failures:
            return f"synced {self.pushed} records to {self.remote}"
        return (
            f"synced {self.pushed} records, {len(self.failures)} sync failures "
            f"(last: {self.failures[-1]}) -- records are still on disk, push manually"
        )

    def _fail(self, message: str) -> bool:
        self.failures.append(message)
        return False

    def _git(self, *args: str) -> _Result:
        try:
            done = subprocess.run(
                ["git", *args],
                cwd=self.root if self.root.is_dir() else Path.cwd(),
                capture_output=True,
                text=True,
                timeout=120,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            return _Result(False, "", str(exc))
        return _Result(done.returncode == 0, done.stdout.strip(), done.stderr.strip())


class _Result:
    def __init__(self, ok: bool, out: str, err: str) -> None:
        self.ok = ok
        self.out = out
        self.err = err
