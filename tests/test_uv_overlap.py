import numpy as np

from lowvram3d.uv_overlap import AREA_EPSILON_UV, positive_area_uv_overlaps


def run(triangles, resolution=2048, **kwargs):
    return positive_area_uv_overlaps(np.asarray(triangles, np.float64), resolution, **kwargs)


def test_separated_triangles_have_no_overlap():
    report = run([
        [[0.10, 0.10], [0.20, 0.10], [0.10, 0.20]],
        [[0.60, 0.60], [0.70, 0.60], [0.60, 0.70]],
    ])
    assert report.success
    assert report.positive_overlap_pair_count == 0
    assert report.positive_overlap_total_texels_equivalent == 0.0


def test_shared_single_vertex_is_not_overlap():
    report = run([
        [[0.10, 0.10], [0.20, 0.10], [0.10, 0.20]],
        [[0.20, 0.10], [0.30, 0.10], [0.30, 0.20]],
    ])
    assert report.success
    assert report.positive_overlap_pair_count == 0


def test_shared_complete_edge_is_not_overlap():
    report = run([
        [[0.10, 0.10], [0.30, 0.10], [0.10, 0.30]],
        [[0.30, 0.10], [0.10, 0.30], [0.30, 0.30]],
    ])
    assert report.success
    assert report.positive_overlap_pair_count == 0


def test_identical_triangles_overlap():
    triangle = [[0.10, 0.10], [0.30, 0.10], [0.10, 0.30]]
    report = run([triangle, triangle])
    assert report.success
    assert report.positive_overlap_pair_count == 1
    assert report.positive_overlap_total_area_uv > AREA_EPSILON_UV


def test_partially_intersecting_triangles_overlap():
    report = run([
        [[0.10, 0.10], [0.40, 0.10], [0.10, 0.40]],
        [[0.20, 0.20], [0.50, 0.20], [0.20, 0.50]],
    ])
    assert report.success
    assert report.positive_overlap_pair_count == 1


def test_triangle_fully_inside_another_overlaps():
    report = run([
        [[0.10, 0.10], [0.90, 0.10], [0.10, 0.90]],
        [[0.20, 0.20], [0.30, 0.20], [0.20, 0.30]],
    ])
    assert report.success
    assert report.positive_overlap_pair_count == 1


def test_adjacent_triangles_folded_over_each_other_are_detected():
    # Shares the edge (0.1,0.1)-(0.3,0.1) but folds back across it instead of away.
    report = run([
        [[0.10, 0.10], [0.30, 0.10], [0.20, 0.30]],
        [[0.10, 0.10], [0.30, 0.10], [0.20, 0.25]],
    ])
    assert report.success
    assert report.positive_overlap_pair_count == 1, "folded neighbours must not be excluded"


def test_degenerate_uv_triangle_is_rejected():
    report = run([
        [[0.10, 0.10], [0.20, 0.10], [0.30, 0.10]],
        [[0.60, 0.60], [0.70, 0.60], [0.60, 0.70]],
    ])
    assert report.degenerate_uv_triangle_count == 1


def test_uv_outside_atlas_is_rejected():
    report = run([
        [[1.20, 0.10], [1.40, 0.10], [1.20, 0.30]],
        [[0.60, 0.60], [0.70, 0.60], [0.60, 0.70]],
    ])
    assert report.out_of_bounds_triangle_count == 1


def test_candidate_pair_cap_fails_closed():
    rng = np.random.default_rng(3)
    base = rng.uniform(0.2, 0.4, size=(400, 1, 2))
    triangles = np.concatenate(
        [base, base + [[0.01, 0.0]], base + [[0.0, 0.01]]], axis=1
    )
    report = run(triangles, max_candidate_pairs=10)
    assert not report.success
    assert any("candidate pair count exceeded" in error for error in report.errors)


def test_timeout_fails_closed():
    rng = np.random.default_rng(5)
    base = rng.uniform(0.2, 0.4, size=(400, 1, 2))
    triangles = np.concatenate(
        [base, base + [[0.01, 0.0]], base + [[0.0, 0.01]]], axis=1
    )
    report = run(triangles, timeout_seconds=0.0)
    assert not report.success
    assert report.timed_out


def test_noise_intersections_are_recorded_not_counted():
    # Overlap far below AREA_EPSILON_UV must be ignored rather than failing the gate.
    tiny = AREA_EPSILON_UV * 0.01
    report = run([
        [[0.10, 0.10], [0.30, 0.10], [0.10, 0.30]],
        [[0.30 - tiny, 0.10], [0.50, 0.10], [0.50, 0.30]],
    ])
    assert report.positive_overlap_pair_count == 0
