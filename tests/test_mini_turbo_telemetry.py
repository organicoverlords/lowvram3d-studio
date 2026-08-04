import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import workers.mini_turbo_generate as telemetry
from workers.mini_turbo_generate import (
    CONSOLE_DEGRADED,
    CONSOLE_HEALTHY,
    _cuda_stats,
    _mesh_summary,
    _mesh_boundary_summary,
    _reset_console_state,
    _safe_console_write,
    _sanitize_decoded_mesh,
    _tensor_summary,
    _trace,
    _write_json_artifact,
)


class SyntheticMesh:
    vertices = [0, 1, 2, 3]
    faces = [(0, 1, 2), (0, 2, 3)]


class SyntheticDecodedMesh:
    def __init__(self):
        self.vertices = np.asarray(
            [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)],
            dtype=np.float32,
        )
        self.faces = np.asarray([(0, 0, 1), (0, 1, 2), (0, 0, 3)], dtype=np.int64)

    def update_faces(self, mask):
        self.faces = self.faces[mask]

    def remove_unreferenced_vertices(self):
        used = np.unique(self.faces)
        remap = np.full(len(self.vertices), -1, dtype=np.int64)
        remap[used] = np.arange(len(used))
        self.vertices = self.vertices[used]
        self.faces = remap[self.faces]


class FailingStream(io.StringIO):
    def __init__(self, error):
        super().__init__()
        self.error = error

    def write(self, value):
        self.write_count = getattr(self, "write_count", 0) + 1
        raise self.error


class FlushFailingStream(io.StringIO):
    def flush(self):
        raise OSError(22, "invalid argument")


class InspectingStream(io.StringIO):
    def __init__(self, callback):
        super().__init__()
        self.callback = callback

    def write(self, value):
        self.callback()
        return super().write(value)


class FakeCuda:
    def __init__(self):
        self.synchronize_calls = 0

    def is_available(self):
        return True

    def synchronize(self):
        self.synchronize_calls += 1

    def memory_allocated(self):
        return 10

    def memory_reserved(self):
        return 20

    def max_memory_allocated(self):
        return 30


class FakeTorch:
    def __init__(self):
        self.cuda = FakeCuda()


class FakeScalar:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value


class FakeFinite:
    def all(self):
        return FakeScalar(True)


class FakeTensor:
    is_cuda = False
    shape = (2,)
    dtype = "torch.float32"
    device = "cpu"

    def detach(self):
        return self

    def numel(self):
        return 2

    def float(self):
        return self

    def amin(self):
        return FakeScalar(1.0)

    def amax(self):
        return FakeScalar(2.0)


class FakeTensorModule(FakeTorch):
    Tensor = FakeTensor

    @staticmethod
    def isfinite(value):
        return FakeFinite()


class MiniTurboTelemetryTests(unittest.TestCase):
    def setUp(self):
        _reset_console_state()

    def tearDown(self):
        _reset_console_state()

    def test_stdout_oserror_keeps_trace_and_calls_artifact_callback(self):
        with tempfile.TemporaryDirectory() as temp:
            trace = Path(temp) / "events.jsonl"
            callback_records = []
            with mock.patch("workers.mini_turbo_generate.sys.stdout", FailingStream(OSError(22, "sink"))):
                record = _trace(trace, "synthetic_mesh", artifact_callback=callback_records.append, **_mesh_summary(SyntheticMesh()))
            sink_event = json.loads(trace.read_text().splitlines()[1])
            self.assertEqual(sink_event["operation"], "console_sink_failure")
            self.assertEqual(sink_event["error"]["errno"], 22)
            self.assertEqual(callback_records[0]["triangles"], 2)
            self.assertEqual(json.loads(trace.read_text().splitlines()[0])["operation"], "synthetic_mesh")

    def test_durable_event_is_written_before_console_is_invoked(self):
        with tempfile.TemporaryDirectory() as temp:
            trace = Path(temp) / "events.jsonl"
            observed = []

            def inspect_durable_event():
                observed.append(trace.exists() and trace.read_text().count("\n") == 1)

            with mock.patch("workers.mini_turbo_generate.sys.stdout", InspectingStream(inspect_durable_event)):
                _trace(trace, "ordered_event")
            self.assertEqual(observed, [True])

    def test_degraded_console_is_not_retried_by_repeated_trace_calls(self):
        with tempfile.TemporaryDirectory() as temp:
            stream = FailingStream(OSError(22, "sink"))
            with mock.patch("workers.mini_turbo_generate.sys.stdout", stream):
                _trace(Path(temp) / "events.jsonl", "first")
                _trace(Path(temp) / "events.jsonl", "second")
            self.assertEqual(stream.write_count, 1)
            self.assertEqual(telemetry._console_state, CONSOLE_DEGRADED)

    def test_unicode_failure_is_non_fatal_and_degrades_console(self):
        error = _safe_console_write(FailingStream(UnicodeError("encoding")), "event")
        self.assertEqual(error["type"], "UnicodeError")

    def test_console_state_names_are_explicit(self):
        self.assertEqual(CONSOLE_HEALTHY, "HEALTHY")
        self.assertEqual(CONSOLE_DEGRADED, "DEGRADED")

    def test_flush_failure_is_recorded_after_write(self):
        stream = FlushFailingStream()
        error = _safe_console_write(stream, "event\n")
        self.assertEqual(error["errno"], 22)
        self.assertEqual(stream.getvalue(), "event\n")

    def test_stdout_and_stderr_failures_are_independent(self):
        stdout_error = _safe_console_write(FailingStream(OSError(22, "stdout")), "out\n")
        _reset_console_state()
        stderr_error = _safe_console_write(FailingStream(OSError(5, "stderr")), "err\n")
        self.assertEqual(stdout_error["errno"], 22)
        self.assertEqual(stderr_error["errno"], 5)

    def test_synthetic_mesh_summary_is_serializable(self):
        self.assertEqual(_mesh_summary(SyntheticMesh()), {"vertices": 4, "triangles": 2})

    def test_decoder_sanitizer_removes_only_duplicate_index_faces(self):
        mesh = SyntheticDecodedMesh()
        before = _mesh_boundary_summary(mesh)
        mesh, removed = _sanitize_decoded_mesh(mesh)
        after = _mesh_boundary_summary(mesh)
        self.assertEqual(before["duplicate_index_faces"], 2)
        self.assertEqual(removed, 2)
        self.assertEqual(after["triangles"], 1)
        self.assertEqual(after["duplicate_index_faces"], 0)
        self.assertNotEqual(before["index_array_sha256"], after["index_array_sha256"])

    def test_valid_status_events_and_error_json_after_sink_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            status = root / "status.json"
            events = root / "events.jsonl"
            error = root / "error.json"
            with mock.patch("workers.mini_turbo_generate.sys.stdout", FailingStream(OSError(22, "sink"))):
                _trace(events, "failure", artifact_callback=lambda record: _write_json_artifact(status, {"status": "failed", "event": record}))
            _write_json_artifact(error, {"error": "synthetic failure", "status": "failed"})
            self.assertEqual(json.loads(status.read_text())["status"], "failed")
            self.assertEqual(json.loads(error.read_text())["status"], "failed")
            sink_event = json.loads(events.read_text().splitlines()[1])
            self.assertEqual(sink_event["operation"], "console_sink_failure")
            self.assertEqual(sink_event["error"]["errno"], 22)

    def test_production_cuda_stats_does_not_synchronize(self):
        fake = FakeTorch()
        _cuda_stats(fake, synchronize=False)
        self.assertEqual(fake.cuda.synchronize_calls, 0)

    def test_diagnostic_cuda_stats_synchronizes_at_named_boundary_only(self):
        fake = FakeTorch()
        _cuda_stats(fake, synchronize=True, boundary_name="named_boundary")
        self.assertEqual(fake.cuda.synchronize_calls, 1)
        with self.assertRaises(ValueError):
            _cuda_stats(fake, synchronize=True)
        self.assertEqual(fake.cuda.synchronize_calls, 1)

    def test_production_tensor_summary_has_no_finite_min_max_scan(self):
        with mock.patch.dict("sys.modules", {"torch": FakeTensorModule()}):
            summary = _tensor_summary("cpu_tensor", FakeTensor(), diagnostic=False)
        self.assertEqual(summary["shape"], [2])
        self.assertNotIn("finite", summary)
        self.assertNotIn("min", summary)
        self.assertNotIn("max", summary)

    def test_diagnostic_tensor_summary_may_scan_finite_min_max(self):
        with mock.patch.dict("sys.modules", {"torch": FakeTensorModule()}):
            summary = _tensor_summary("cpu_tensor", FakeTensor(), diagnostic=True)
        self.assertTrue(summary["finite"])
        self.assertEqual(summary["min"], 1.0)
        self.assertEqual(summary["max"], 2.0)


if __name__ == "__main__":
    unittest.main()


class ConditioningDimensionsTest(unittest.TestCase):
    """The multiview path rebinds `conditioning` from an image to a dict.

    Telemetry read `.size` off it regardless, which killed every multiview run
    one statement before the pipeline was built. The failure surfaced as
    "'dict' object has no attribute 'size'" and was misread twice as an
    hy3dgen incompatibility.
    """

    class FakeImage:
        def __init__(self, size):
            self.size = size

    def test_single_image_reports_a_flat_pair(self):
        image = self.FakeImage((950, 950))
        self.assertEqual(
            telemetry._conditioning_dimensions(image), [950, 950])

    def test_view_dict_reports_a_size_per_named_view(self):
        views = {"front": self.FakeImage((950, 950)),
                 "left": self.FakeImage((512, 512))}
        self.assertEqual(
            telemetry._conditioning_dimensions(views),
            {"front": [950, 950], "left": [512, 512]})

    def test_view_dict_is_json_serialisable(self):
        # It is written into a JSONL trace, so a tuple-keyed or tuple-valued
        # result would fail at the point of recording rather than here.
        views = {"front": self.FakeImage((950, 950))}
        json.dumps(telemetry._conditioning_dimensions(views))
