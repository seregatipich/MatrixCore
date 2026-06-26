"""Smoke tests that every example and the benchmark stay runnable."""

import runpy
from pathlib import Path

import numpy as np
import pytest

EXAMPLES = sorted((Path(__file__).resolve().parent.parent / "examples").glob("*.py"))


@pytest.mark.parametrize("example", EXAMPLES, ids=lambda p: p.name)
def test_example_runs(example, capsys):
    runpy.run_path(str(example), run_name="__main__")
    assert capsys.readouterr().out.strip()


def test_benchmark_helpers():
    from benchmarks.benchmark import spd, timed

    A = spd(8)
    assert np.allclose(A, A.T)
    assert np.all(np.linalg.eigvalsh(A) > 0)

    result, elapsed = timed(lambda: sum(range(100)), repeats=2)
    assert result == 4950
    assert elapsed >= 0.0
