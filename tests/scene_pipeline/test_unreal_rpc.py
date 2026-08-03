from __future__ import annotations

import json
import struct

import pytest

from workers.scene_pipeline.unreal_rpc import (
    UnrealRPCClient,
    UnrealRPCError,
    decode_frame,
    encode_frame,
)


class FakeSocket:
    def __init__(self, incoming: bytes = b"") -> None:
        self.incoming = bytearray(incoming)
        self.sent: list[bytes] = []
        self.timeout = None

    def settimeout(self, value: float) -> None:
        self.timeout = value

    def sendall(self, data: bytes) -> None:
        self.sent.append(data)

    def recv(self, size: int) -> bytes:
        if not self.incoming:
            return b""
        chunk = bytes(self.incoming[: max(1, min(size, 3))])
        del self.incoming[: len(chunk)]
        return chunk

    def close(self) -> None:
        pass


def response_frame(request_id: int, result: object) -> bytes:
    return encode_frame({"jsonrpc": "2.0", "id": request_id, "result": result})


def test_frame_is_big_endian_utf8_json() -> None:
    frame = encode_frame({"message": "å", "ok": True})
    assert struct.unpack(">I", frame[:4])[0] == len(frame) - 4
    assert json.loads(frame[4:].decode("utf-8")) == {"message": "å", "ok": True}


def test_decode_handles_fragmented_reads() -> None:
    message = {"jsonrpc": "2.0", "id": 4, "result": {"ok": True}}
    assert decode_frame(FakeSocket(encode_frame(message))) == message


def test_client_correlates_ids_and_calls_initialize() -> None:
    sock = FakeSocket(response_frame(1, {"protocolVersion": "2025-11-25"}))
    client = UnrealRPCClient(socket_factory=lambda *_: sock)
    assert client.initialize()["protocolVersion"] == "2025-11-25"
    sent = json.loads(sock.sent[0][4:].decode("utf-8"))
    assert sent["jsonrpc"] == "2.0"
    assert sent["method"] == "initialize"
    assert sent["id"] == 1


def test_client_rejects_mismatched_response_id() -> None:
    sock = FakeSocket(response_frame(9, {"ok": True}))
    client = UnrealRPCClient(socket_factory=lambda *_: sock)
    with pytest.raises(UnrealRPCError, match="correlation"):
        client.call("health_check")


def test_client_surfaces_json_rpc_error() -> None:
    sock = FakeSocket(encode_frame({"jsonrpc": "2.0", "id": 1, "error": {"code": -1}}))
    client = UnrealRPCClient(socket_factory=lambda *_: sock)
    with pytest.raises(UnrealRPCError, match="failed"):
        client.call("health_check")


def test_oversized_frame_rejected() -> None:
    with pytest.raises(UnrealRPCError, match="outside"):
        encode_frame({"x": "a" * 32}, max_bytes=8)
