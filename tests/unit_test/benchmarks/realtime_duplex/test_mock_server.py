# SPDX-License-Identifier: Apache-2.0
"""Mock-server protocol checks and driver session-log round-trip.

Stage-2 coverage for the harness: the scripted mock server speaks the
conversation protocol subset correctly (positive flow plus the three
rejection paths), and the driver records a session log that survives a
write/read round-trip. Verdict-level checks land with replay in stage 3.
Pure CPU; no model or GPU.
"""

from __future__ import annotations

import asyncio
import base64
import json

import pytest
import websockets

from benchmarks.realtime_duplex import schema
from benchmarks.realtime_duplex.harness import DuplexSessionDriver
from benchmarks.realtime_duplex.mock_server import (
    TONE_BYTE,
    DuplexMockServer,
    MockBehavior,
)
from benchmarks.realtime_duplex.session_log import read_session_log


def make_audio(frames: int) -> bytes:
    return b"\x55\xaa" * (schema.frame_bytes() * frames // 2)


def drive(behavior: MockBehavior, **driver_kwargs: object) -> object:
    async def once() -> object:
        async with DuplexMockServer(behavior) as server:
            driver = DuplexSessionDriver(server.url, make_audio(6), **driver_kwargs)
            return await driver.run()

    return asyncio.run(once())


def speak_with_mock(behavior: MockBehavior, client_script: object) -> list[dict]:
    """Run a raw websocket client script against the mock; return its events."""

    async def once() -> list[dict]:
        events: list[dict] = []
        async with DuplexMockServer(behavior) as server:
            async with websockets.connect(server.url, max_size=None) as ws:

                async def next_event(timeout_s: float = 5.0) -> dict:
                    raw = await asyncio.wait_for(ws.recv(), timeout_s)
                    event = json.loads(raw)
                    events.append(event)
                    return event

                await client_script(ws, next_event)
                # Absorb anything still in flight so `events` is complete.
                try:
                    while True:
                        await next_event(0.2)
                except (asyncio.TimeoutError, websockets.exceptions.ConnectionClosed):
                    pass
        return events

    return asyncio.run(once())


def types_of(events: list[dict]) -> list[str]:
    return [str(event.get("type", "")) for event in events]


def test_mock_server_speaks_scripted_response_flow() -> None:
    async def script(ws: object, next_event: object) -> None:
        assert (await next_event())["type"] == "session.created"
        await ws.send(json.dumps({"type": "session.update", "session": {}}))
        assert (await next_event())["type"] == "session.updated"
        payload = base64.b64encode(b"\x00" * schema.frame_bytes()).decode("ascii")
        for _ in range(3):
            await ws.send(
                json.dumps({"type": "input_audio_buffer.append", "audio": payload})
            )
        for expected in (
            "input_audio_buffer.speech_started",
            "input_audio_buffer.speech_stopped",
            "input_audio_buffer.committed",
            "conversation.item.created",
            "response.created",
        ):
            assert (await next_event())["type"] == expected
        deltas = [await next_event() for _ in range(3)]
        for delta in deltas:
            assert delta["type"] == "response.audio.delta"
            audio = base64.b64decode(delta["delta"])
            assert audio == TONE_BYTE.to_bytes() * schema.frame_bytes()
        done = await next_event()
        assert done["type"] == "response.audio.done"
        assert done["output_audio_bytes"] == 3 * schema.frame_bytes()
        finished = await next_event()
        assert finished["type"] == "response.done"
        assert finished["response"]["status"] == "completed"

    events = speak_with_mock(MockBehavior(), script)
    assert types_of(events).count("response.audio.delta") == 3


def test_mock_server_rejects_bad_client_events() -> None:
    async def script(ws: object, next_event: object) -> None:
        assert (await next_event())["type"] == "session.created"
        await ws.send("this-line-is-not-json")
        assert (await next_event())["error"]["message"] == "malformed JSON"
        await ws.send("[1, 2, 3]")
        assert (await next_event())["error"]["message"] == "event is not an object"
        await ws.send(json.dumps({"type": "bogus.probe"}))
        error = await next_event()
        assert error["type"] == "error"
        assert "bogus.probe" in error["error"]["message"]

    events = speak_with_mock(MockBehavior(), script)
    assert types_of(events).count("error") == 3


def test_driver_continuous_session_log_roundtrip(tmp_path: object) -> None:
    record = drive(MockBehavior(), pace=0.1, model="mock-duplex")
    assert record.client_error is None

    s2c = [e for e in record.entries if e.direction == "s2c"]
    c2s = [e for e in record.entries if e.direction == "c2s"]
    assert sum(e.type == "response.audio.delta" for e in s2c) == 3
    appends = [e for e in c2s if e.type == "input_audio_buffer.append"]
    assert len(appends) == 6
    assert all(e.audio_bytes == schema.frame_bytes() for e in appends)
    assert any(e.type == "response.done" for e in s2c)

    assert record.manifest.scenario == "continuous"
    assert record.manifest.model == "mock-duplex"
    assert record.manifest.input_duration_s == pytest.approx(6 * 0.08)
    assert record.manifest.pace == 0.1

    path = tmp_path / "session.jsonl"
    record.save(str(path))
    manifest, entries = read_session_log(str(path))
    assert manifest.scenario == "continuous"
    assert len(entries) == len(record.entries)


def test_driver_records_malformed_server_line() -> None:
    record = drive(MockBehavior(send_malformed_on_session=True))
    assert record.client_error is None
    assert any(e.type == "malformed.line" for e in record.entries)


def test_driver_validates_constructor_arguments() -> None:
    audio = make_audio(1)
    for kwargs in ({"mode": "nope"}, {"pace": 0}, {"pace": -1.0}):
        with pytest.raises(ValueError):
            DuplexSessionDriver("ws://127.0.0.1:1", audio, **kwargs)
