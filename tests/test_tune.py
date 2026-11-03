from __future__ import annotations

from pathlib import Path

import pytest

from stockpred.tune import SearchSpace, TuningError, run_search


def _quadratic(params: dict) -> float:
    """Peak at (x=7, y=2.5); cheap and deterministic so the search is the only cost."""
    return -((params["x"] - 7) ** 2) - ((params["y"] - 2.5) ** 2)


SPACE = SearchSpace(ints={"x": (1, 10)}, floats={"y": (0.0, 5.0)})


def test_search_finds_the_optimum() -> None:
    result = run_search(_quadratic, SPACE, study_name="quad", n_trials=25, seed=3)
    assert result.best_value > -0.5
    assert abs(result.best_params["x"] - 7) <= 2
    assert result.n_trials >= 1


def test_search_is_reproducible_with_seed() -> None:
    first = run_search(_quadratic, SPACE, study_name="a", n_trials=10, seed=11)
    second = run_search(_quadratic, SPACE, study_name="b", n_trials=10, seed=11)
    assert first.best_params == second.best_params


def test_storage_persists_and_resumes(tmp_path: Path) -> None:
    db = tmp_path / "studies" / "tune.db"
    first = run_search(_quadratic, SPACE, study_name="persist", n_trials=5, seed=1, storage=db)
    second = run_search(_quadratic, SPACE, study_name="persist", n_trials=5, seed=1, storage=db)

    assert db.is_file()
    assert first.n_trials == 5
    assert second.n_trials == 10  # resumed, not restarted
    assert second.storage == str(db)


def test_minimize_direction() -> None:
    result = run_search(
        lambda p: -_quadratic(p),
        SPACE,
        study_name="min",
        n_trials=10,
        seed=5,
        direction="minimize",
    )
    assert result.best_value < 0.5


def test_empty_space_rejected() -> None:
    with pytest.raises(TuningError, match="empty"):
        run_search(_quadratic, SearchSpace(), study_name="x")


def test_bad_trial_count_rejected() -> None:
    with pytest.raises(TuningError, match="n_trials"):
        run_search(_quadratic, SPACE, study_name="x", n_trials=0)


def test_bad_direction_rejected() -> None:
    with pytest.raises(TuningError, match="direction"):
        run_search(_quadratic, SPACE, study_name="x", direction="sideways")


def test_inverted_range_rejected() -> None:
    with pytest.raises(TuningError, match="must be increasing"):
        run_search(_quadratic, SearchSpace(ints={"x": (10, 1)}), study_name="x")


def test_nan_objective_is_pruned_not_counted(tmp_path: Path) -> None:
    def flaky(params: dict) -> float:
        if params["x"] > 5:
            return float("nan")
        return -float(params["x"])

    result = run_search(
        flaky, SearchSpace(ints={"x": (1, 10)}), study_name="nan", n_trials=10, seed=2
    )
    assert result.best_params["x"] <= 5


def test_result_is_serialisable() -> None:
    result = run_search(_quadratic, SPACE, study_name="ser", n_trials=3, seed=9)
    payload = result.as_dict()
    assert set(payload) == {"study_name", "best_params", "best_value", "n_trials", "storage"}
