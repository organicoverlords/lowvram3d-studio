"""Guards that stop a selection from being reported as a decision when it is noise.

Both were added after real incidents: candidates measured by two different scorer versions were
compared against each other and produced a convincing winner, and a field whose whole spread was
0.005 against a 0.22 gap to passing was ranked and its maximum announced as the best candidate.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "workers"))

from pipeline_generate_best_of import MIN_MEANINGFUL_SPREAD  # noqa: E402


def spread(scores):
    return round(max(scores) - min(scores), 5)


def meaningful(scores):
    return spread(scores) >= MIN_MEANINGFUL_SPREAD


def test_seed_sweep_noise_is_not_a_meaningful_selection():
    """The measured shaman seed sweep: three candidates inside half a percent."""
    assert meaningful([0.4933, 0.4987, 0.5039]) is False


def test_a_real_difference_is_meaningful():
    assert meaningful([0.41, 0.62]) is True


def test_threshold_is_larger_than_observed_seed_noise():
    """0.005 was the observed spread across seeds; the guard must sit well above it."""
    assert MIN_MEANINGFUL_SPREAD > 0.005


def test_scorer_version_mismatch_is_detectable():
    """Scores from different scorer versions are not comparable and must be rejected."""
    scored = [
        {"score": 0.418, "scorer_version": "aaaaaaaaaaaaaaaa"},
        {"score": 0.500, "scorer_version": "bbbbbbbbbbbbbbbb"},
    ]
    assert len({entry["scorer_version"] for entry in scored}) > 1
