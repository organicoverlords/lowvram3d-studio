from __future__ import annotations

import math

import pytest


torch = pytest.importorskip("torch")

from workers.mv_adapter_numeric_probe import (
    NO_NONFINITE_DECISION,
    STATISTIC_FIELDS,
    FirstNonfiniteTensor,
    NumericProbe,
    classify_boundary,
    decide_next_action,
    tensor_statistics,
)


def test_statistics_expose_every_required_field():
    statistics = tensor_statistics(torch.zeros((2, 3)))
    assert tuple(statistics.keys()) == STATISTIC_FIELDS


def test_finite_statistics_match_known_values():
    tensor = torch.tensor([[-2.0, 0.0], [1.0, 5.0]], dtype=torch.float32)
    statistics = tensor_statistics(tensor)

    assert statistics["shape"] == [2, 2]
    assert statistics["dtype"] == "torch.float32"
    assert statistics["device"] == "cpu"
    assert statistics["finite_count"] == 4
    assert statistics["nonfinite_count"] == 0
    assert statistics["finite_ratio"] == 1.0
    assert statistics["minimum"] == pytest.approx(-2.0)
    assert statistics["maximum"] == pytest.approx(5.0)
    assert statistics["mean"] == pytest.approx(1.0)
    assert statistics["standard_deviation"] == pytest.approx(2.5495097, rel=1e-6)
    assert statistics["absolute_maximum"] == pytest.approx(5.0)


def test_statistics_ignore_nonfinite_elements_but_count_them():
    tensor = torch.tensor(
        [1.0, float("nan"), float("inf"), float("-inf"), -3.0], dtype=torch.float32
    )
    statistics = tensor_statistics(tensor)

    assert statistics["finite_count"] == 2
    assert statistics["nonfinite_count"] == 3
    assert statistics["finite_ratio"] == pytest.approx(2 / 5)
    assert statistics["minimum"] == pytest.approx(-3.0)
    assert statistics["maximum"] == pytest.approx(1.0)
    assert statistics["mean"] == pytest.approx(-1.0)
    assert statistics["absolute_maximum"] == pytest.approx(3.0)
    assert all(
        statistics[field] is not None and not math.isnan(statistics[field])
        for field in ("minimum", "maximum", "mean", "absolute_maximum")
    )


def test_statistics_are_none_when_no_element_is_finite():
    statistics = tensor_statistics(torch.full((3,), float("nan")))

    assert statistics["finite_count"] == 0
    assert statistics["nonfinite_count"] == 3
    assert statistics["finite_ratio"] == 0.0
    for field in ("minimum", "maximum", "mean", "standard_deviation", "absolute_maximum"):
        assert statistics[field] is None


def test_single_element_standard_deviation_is_zero_not_nan():
    statistics = tensor_statistics(torch.tensor([4.0]))
    assert statistics["standard_deviation"] == pytest.approx(0.0)


def test_half_precision_statistics_are_computed_in_float32():
    tensor = torch.tensor([1.5, -2.5], dtype=torch.float16)
    statistics = tensor_statistics(tensor)

    assert statistics["dtype"] == "torch.float16"
    assert statistics["minimum"] == pytest.approx(-2.5)
    assert statistics["absolute_maximum"] == pytest.approx(2.5)


def test_statistics_reject_non_tensor_input():
    with pytest.raises(TypeError):
        tensor_statistics([1.0, 2.0])


def test_probe_records_finite_tensors_in_order():
    probe = NumericProbe()
    probe.record("reference_latents", torch.zeros((1, 4, 8, 8)))
    probe.record("initial_noise_latents", torch.ones((6, 4, 8, 8)), step=None)

    assert probe.first_nonfinite is None
    assert [entry["order"] for entry in probe.records] == [1, 2]
    assert probe.summary()["probed_labels"] == [
        "reference_latents",
        "initial_noise_latents",
    ]
    assert probe.records[1]["step"] is None


def test_probe_fails_closed_on_the_first_nonfinite_tensor():
    probe = NumericProbe()
    probe.record("reference_latents", torch.zeros((1, 4)))

    with pytest.raises(FirstNonfiniteTensor) as excinfo:
        probe.record(
            "unet_noise_pred_step_00", torch.tensor([0.0, float("inf")]), step=0
        )

    assert excinfo.value.label == "unet_noise_pred_step_00"
    assert probe.first_nonfinite_label == "unet_noise_pred_step_00"
    assert probe.first_nonfinite["statistics"]["nonfinite_count"] == 1
    assert probe.first_nonfinite["step"] == 0


def test_gate_retains_the_failing_record_for_the_report():
    probe = NumericProbe()
    with pytest.raises(FirstNonfiniteTensor):
        probe.record("vae_decode_output", torch.full((2, 2), float("nan")))

    summary = probe.summary()
    assert summary["nonfinite_boundary_found"] is True
    assert summary["probe_record_count"] == 1
    assert summary["records"][0]["statistics"]["finite_ratio"] == 0.0


def test_first_nonfinite_boundary_is_not_overwritten_by_later_failures():
    probe = NumericProbe()
    with pytest.raises(FirstNonfiniteTensor):
        probe.record("vae_decode_input", torch.tensor([float("nan")]))
    with pytest.raises(FirstNonfiniteTensor):
        probe.record("vae_decode_output", torch.tensor([float("nan")]))

    assert probe.first_nonfinite_label == "vae_decode_input"


@pytest.mark.parametrize(
    ("label", "category"),
    [
        ("reference_latents", "REFERENCE_LATENTS"),
        ("control_image_prepared", "CONTROL_ENCODER_INPUT"),
        ("condition_encoder_input", "CONTROL_ENCODER_INPUT"),
        ("adapter_state_02", "CONTROL_ENCODER_OUTPUT"),
        ("initial_noise_latents", "INITIAL_NOISE"),
        ("unet_noise_pred_step_03", "UNET_OUTPUT"),
        ("reference_unet_output", "UNET_OUTPUT"),
        ("scheduler_latents_step_03", "SCHEDULER_LATENTS"),
        ("final_pre_decode_latents", "SCHEDULER_LATENTS"),
        ("vae_decode_input", "VAE_DECODE_INPUT"),
        ("vae_decode_output", "VAE_DECODE_OUTPUT"),
        ("something_else", "UNCLASSIFIED"),
    ],
)
def test_boundary_classification(label, category):
    assert classify_boundary(label) == category


@pytest.mark.parametrize(
    ("label", "decision"),
    [
        (
            "unet_noise_pred_step_00",
            "STOP_PLAIN_I2MV_ROUTE_INSPECT_OFFICIAL_GEOMETRY_GUIDED_SD21_MV_ADAPTER",
        ),
        (
            "scheduler_latents_step_00",
            "STOP_PLAIN_I2MV_ROUTE_INSPECT_OFFICIAL_GEOMETRY_GUIDED_SD21_MV_ADAPTER",
        ),
        ("vae_decode_input", "REPAIR_VAE_SCALING_DTYPE_BOUNDARY_ONE_CORRECTED_RETRY"),
        ("vae_decode_output", "REPAIR_VAE_SCALING_DTYPE_BOUNDARY_ONE_CORRECTED_RETRY"),
        (
            "adapter_state_00",
            "INSPECT_CONTROL_IMAGE_NORMALISATION_AND_CONDITION_ENCODER_PRECISION",
        ),
    ],
)
def test_decisions_follow_the_task_rules(label, decision):
    verdict = decide_next_action(label)
    assert verdict["decision"] == decision
    assert verdict["decision_from_task_rules"] is True


def test_all_finite_decision_targets_postprocessing_only():
    verdict = decide_next_action(None)
    assert verdict["boundary_category"] == "NONE_ALL_PROBED_TENSORS_FINITE"
    assert verdict["decision"] == NO_NONFINITE_DECISION[0]


def test_reference_latent_decision_is_flagged_as_an_extension():
    verdict = decide_next_action("reference_latents")
    assert verdict["decision_from_task_rules"] is False
