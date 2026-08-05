"""Source/model pairing must refuse evidence it should not accept.

Each case here is a false pairing this discovery worker actually produced: outputs mistaken for
sources, one document's single source spread across every model it happened to mention, and the
worker's own manifest re-read as evidence.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "workers"))

from benchmark_discover import (  # noqa: E402
    FORBIDDEN_BASENAMES,
    SELF_REPORT_NAMES,
    find_source_reference,
    scoped_source_reference,
)


def index(*names):
    return {name.lower(): [Path("C:/refs") / name] for name in names}


def test_derived_artefacts_are_never_treated_as_sources():
    """A receipt naming its own normal map is describing an output, not an input."""
    for artefact in ("shaman_normal_4k.png", "asset_basecolor_4k.png", "debug_overlay.png",
                     "state_orm_4k.png", "matte.png"):
        document = {"source_image": artefact}
        assert find_source_reference(document, index(artefact)) is None


def test_a_real_source_reference_is_accepted():
    document = {"source_image": "concept_art.png"}
    found = find_source_reference(document, index("concept_art.png"))
    assert found is not None and found[1].name == "concept_art.png"


def test_non_source_keys_are_ignored():
    """Only explicitly source-named keys count; a bare image string is not evidence."""
    assert find_source_reference({"preview": "concept_art.png"}, index("concept_art.png")) is None
    assert find_source_reference({"notes": ["concept_art.png"]}, index("concept_art.png")) is None


def test_one_source_does_not_pair_with_every_model_in_the_document():
    """A manifest listing many meshes and one source must not pair all of them to it."""
    document = {
        "source_image": "concept_art.png",
        "entries": [{"model": "alpha.glb"}, {"model": "beta.glb"}],
    }
    # 'alpha.glb' is mentioned only in a record carrying no source of its own.
    assert scoped_source_reference(document, "alpha.glb", index("concept_art.png")) is None


def test_scoped_pairing_accepts_a_source_in_the_model_s_own_record():
    document = {"entries": [
        {"model": "alpha.glb", "source_image": "alpha_concept.png"},
        {"model": "beta.glb", "source_image": "beta_concept.png"},
    ]}
    found = scoped_source_reference(
        document, "alpha.glb", index("alpha_concept.png", "beta_concept.png"))
    assert found is not None and found[1].name == "alpha_concept.png"


def test_self_reports_and_prohibited_assets_are_declared():
    assert "benchmark_manifest.json" in SELF_REPORT_NAMES
    assert "shaman.fbx" in FORBIDDEN_BASENAMES
