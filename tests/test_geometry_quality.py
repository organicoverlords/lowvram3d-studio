from lowvram3d.geometry_quality import ComponentMetrics, decide_component, topology_gate


def metrics(**overrides):
    values = dict(
        face_count=10,
        face_fraction=0.0002,
        area_fraction=0.0001,
        extent_fraction=0.01,
        contact_ratio=0.0,
        nearest_distance_fraction=0.1,
        is_main=False,
    )
    values.update(overrides)
    return ComponentMetrics(**values)


def test_main_is_never_removed():
    assert decide_component(metrics(is_main=True), "single_subject_strict").removable is False


def test_conservative_removes_only_tiny_detached_debris():
    decision = decide_component(metrics(), "conservative")
    assert decision.action == "REMOVE_TINY_DEBRIS"
    assert decision.removable is True


def test_conservative_keeps_large_detached_component():
    decision = decide_component(
        metrics(face_count=500, face_fraction=0.02, area_fraction=0.01, extent_fraction=0.2),
        "conservative",
    )
    assert decision.action == "KEEP_AMBIGUOUS_DETACHED"
    assert decision.removable is False


def test_conservative_keeps_attached_tiny_component():
    assert decide_component(metrics(contact_ratio=0.2), "conservative").removable is False


def test_strict_removes_normal_detached_component():
    assert decide_component(metrics(face_count=1500, face_fraction=0.04), "single_subject_strict").removable


def test_strict_protects_major_separate_part():
    decision = decide_component(metrics(face_count=5000, face_fraction=0.1), "single_subject_strict")
    assert decision.action == "KEEP_PROTECTED_MAJOR_PART"
    assert decision.removable is False


def test_strict_removes_stretched_sparse_detached_artifact():
    decision = decide_component(
        metrics(face_count=8, area_fraction=0.0009, extent_fraction=0.53),
        "single_subject_strict",
    )
    assert decision.action == "REMOVE_SPARSE_DETACHED_ARTIFACT"
    assert decision.removable is True


def test_strict_removes_long_rod_artifact_but_keeps_major_part():
    decision = decide_component(
        metrics(face_count=2194, face_fraction=0.003215, area_fraction=0.006544, extent_fraction=0.561),
        "single_subject_strict",
    )
    assert decision.action == "REMOVE_STRETCHED_DETACHED_ARTIFACT"
    assert decision.removable is True


def test_topology_gate_rejects_boundary_explosion():
    passed, errors = topology_gate(
        faces_before=45000,
        faces_after=40000,
        boundary_before=150,
        boundary_after=4900,
        mode="single_subject_strict",
    )
    assert passed is False
    assert any("boundary edges grew" in error for error in errors)


def test_topology_gate_accepts_closed_component_removal():
    passed, errors = topology_gate(
        faces_before=45000,
        faces_after=37000,
        boundary_before=153,
        boundary_after=21,
        mode="single_subject_strict",
    )
    assert passed is True
    assert errors == []
