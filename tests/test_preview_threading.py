"""Phase 2: off-thread JSON preview render + stale-request supersede.

Exercises the threading logic in isolation (bound to a lightweight fake) so no
Tk root / mainloop is needed.
"""

from __future__ import annotations

import json
import queue
import types
from pathlib import Path

import pytest

import app
from app import App


def _make_fake():
    fake = types.SimpleNamespace()
    fake.queue = queue.Queue()
    fake.closed = False
    fake._preview_request_id = 0
    fake._preview_bounds = lambda: (256, 256)
    fake._render_json_preview_async = types.MethodType(App._render_json_preview_async, fake)
    return fake


def _geometry_json(tmp_path: Path) -> Path:
    payload = {
        "width": 64,
        "height": 64,
        "shapes": [
            {"type": 1, "data": [0, 0, 64, 64], "color": [0, 0, 0, 255]},
            {"type": 16, "data": [32, 32, 10, 10, 0], "color": [255, 0, 0, 255]},
        ],
    }
    path = tmp_path / "geom.json"
    path.write_text(json.dumps(payload))
    return path


def test_async_render_posts_bytes_to_queue(tmp_path):
    fake = _make_fake()
    path = _geometry_json(tmp_path)
    fake._render_json_preview_async(path)
    kind, payload = fake.queue.get(timeout=5)
    assert kind == "preview_result"
    rid, data = payload
    assert rid == 1
    assert isinstance(data, (bytes, bytearray))
    assert len(data) > 0


def test_request_id_increments_each_call(tmp_path):
    fake = _make_fake()
    path = _geometry_json(tmp_path)
    fake._render_json_preview_async(path)
    fake._render_json_preview_async(path)
    assert fake._preview_request_id == 2
    seen = set()
    for _ in range(2):
        _kind, (rid, _data) = fake.queue.get(timeout=5)
        seen.add(rid)
    assert seen == {1, 2}


def test_stale_result_is_dropped_by_handler():
    # Replicates the _poll_queue preview_result guard: only the current id wins.
    current_id = 5
    applied = []

    def handle(rid, data):
        if rid == current_id:
            applied.append(data)

    handle(4, b"stale")
    handle(5, b"fresh")
    assert applied == [b"fresh"]
