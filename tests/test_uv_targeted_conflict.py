import numpy as np

from workers.uv_targeted_conflict import diagnose


def test_shared_edge_is_not_a_conflict():
    uv = np.asarray(
        [
            [[0.0, 0.0], [0.5, 0.0], [0.0, 0.5]],
            [[0.5, 0.0], [1.0, 0.0], [1.0, 0.5]],
        ],
        dtype=np.float64,
    )
    report = diagnose(uv, [True, False], [False, True], grid_size=16)
    assert report["success"]
    assert report["positive_overlap_pair_count"] == 0
    assert report["classification"] == "PROVEN"


def test_positive_area_front_rear_conflict_is_proven():
    uv = np.asarray(
        [
            [[0.0, 0.0], [0.8, 0.0], [0.0, 0.8]],
            [[0.2, 0.2], [0.9, 0.2], [0.2, 0.9]],
        ],
        dtype=np.float64,
    )
    report = diagnose(uv, [True, False], [False, True], grid_size=16)
    assert report["success"]
    assert report["positive_overlap_pair_count"] == 1
    assert report["classification"] == "PROVEN"
    assert report["reported_pairs"][0]["front_triangle"] == 0
    assert report["reported_pairs"][0]["rear_triangle"] == 1


def test_candidate_cap_fails_closed():
    uv = np.asarray(
        [
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
        ],
        dtype=np.float64,
    )
    report = diagnose(uv, [True, False], [False, True], grid_size=4, max_candidates=0)
    assert not report["success"]
    assert report["candidate_cap_exceeded"]
    assert report["classification"] == "NOT_PROVEN"
