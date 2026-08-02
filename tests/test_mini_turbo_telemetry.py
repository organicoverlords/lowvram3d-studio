import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from workers.mini_turbo_generate import (
    _mesh_summary,
    _safe_console_write,
    _trace,
    _write_json_artifact,
)


class SyntheticMesh:
    vertices = [0, 1, 2, 3]
    faces = [(0, 1, 2), (0, 2, 3)]


class FailingStream(io.StringIO):
    def __init__(self, error):
        super().__init__()
        self.error = error

    def write(self, value):
        raise self.error


class FlushFailingStream(io.StringIO):
    def flush(self):
        raise OSError(22, "invalid argument")


class MiniTurboTelemetryTests(unittest.TestCase):
    def test_stdout_oserror_keeps_trace_and_calls_artifact_callback(self):
        with tempfile.TemporaryDirectory() as temp:
            trace = Path(temp) / "events.jsonl"
            callback_records = []
            with mock.patch("workers.mini_turbo_generate.sys.stdout", FailingStream(OSError(22, "sink"))):
                record = _trace(trace, "synthetic_mesh", artifact_callback=callback_records.append, **_mesh_summary(SyntheticMesh()))
            self.assertEqual(record["console_sink_error"]["errno"], 22)
            self.assertEqual(callback_records[0]["triangles"], 2)
            self.assertEqual(json.loads(trace.read_text().splitlines()[0])["operation"], "synthetic_mesh")

    def test_flush_failure_is_recorded_after_write(self):
        stream = FlushFailingStream()
        error = _safe_console_write(stream, "event\n")
        self.assertEqual(error["errno"], 22)
        self.assertEqual(stream.getvalue(), "event\n")

    def test_stdout_and_stderr_failures_are_independent(self):
        stdout_error = _safe_console_write(FailingStream(OSError(22, "stdout")), "out\n")
        stderr_error = _safe_console_write(FailingStream(OSError(5, "stderr")), "err\n")
        self.assertEqual(stdout_error["errno"], 22)
        self.assertEqual(stderr_error["errno"], 5)

    def test_synthetic_mesh_summary_is_serializable(self):
        self.assertEqual(_mesh_summary(SyntheticMesh()), {"vertices": 4, "triangles": 2})

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
            self.assertEqual(json.loads(events.read_text().splitlines()[0])["console_sink_error"]["errno"], 22)


if __name__ == "__main__":
    unittest.main()
