"""Record-and-replay (virtual_fly/recording.py, the capture action, the replay endpoints; docs/TWO_FLIES_PLAN.md 8.7-8.8).
The physics parts need flygym (skipped without it); the rest runs on the synthetic connectome in CI."""
import gzip
import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest

from virtual_fly import physics, recording
from virtual_fly import server as S
from virtual_fly.game import Game
from virtual_fly.settings import build_brain

needs_flygym = pytest.mark.skipif(not physics.available(), reason=f"physics body unavailable (optional): {physics._IMPORT_ERROR!r}")


@pytest.fixture
def rec_dir(tmp_path, monkeypatch):
    d = tmp_path / "recordings"
    monkeypatch.setattr(recording, "recordings_dir", lambda: d)
    return d


def test_ids_and_folders_are_plain(rec_dir):
    assert recording.folder_of("../etc") is None and recording.folder_of("20261003-120000/x") is None
    assert recording.folder_of("20261003-120000") is None                 # no such recording
    (rec_dir / "20261003-120000").mkdir(parents=True)
    (rec_dir / "20261003-120000" / "header.json").write_text(json.dumps({"format": 1, "flies": [], "active": False, "ticks": 3}))
    assert recording.folder_of("20261003-120000") == rec_dir / "20261003-120000"
    assert recording.folder_of(42) is None and recording.folder_of("20261003-120000-2") is None
    assert [r["id"] for r in recording.list_recordings()] == ["20261003-120000"]
    (rec_dir / "not-a-recording").mkdir()
    assert len(recording.list_recordings()) == 1


def test_a_drawn_run_is_saved_and_listed(conn, rec_dir):
    g = Game(build_brain(conn, "game", seed=0), seed=1, autopilot=True)
    try:
        assert recording.cannot_start(g) is None
        assert g.action({"type": "capture", "on": False}) == {"ok": False, "error": "no replay is being saved"}
        assert g.action({"type": "capture", "on": True})["ok"]
        for _ in range(6):
            g.tick()
        st = g.state_dict["capture"]
        assert st["active"] and st["ticks"] == 6 and st["physics"] is False and st["error"] is None   # the start lands on tick 1
        rid = st["id"]
        assert g.action({"type": "capture", "on": True}) == {"ok": False, "error": "a replay is being saved already; stop it first"}
        assert g.action({"type": "capture", "on": False})["ok"]
        g.tick()
        assert "capture" not in g.state_dict and g.capture is None      # the key only while saving: the golden frames unchanged
    finally:
        g.close()
    folder = recording.folder_of(rid)
    assert folder is not None and sorted(p.name for p in folder.iterdir()) == ["frames.jsonl.gz", "header.json"]
    h = recording.read_header(folder)
    assert h["format"] == 1 and h["active"] is False and h["ticks"] == 6 and h["seconds"] == 0.15 and h["physics"] is None
    assert h["flies"][0]["body"] == "drawn" and h["tick_ms"] == 25.0 and h["seed"] == 1 and h["bytes"] > 0
    lines = gzip.open(folder / "frames.jsonl.gz", "rt").read().splitlines()
    assert len(lines) == 6
    f = json.loads(lines[-1])
    assert f["seq"] == 6 and f["t"] == 0.15 and len(f["flies"]) == 1
    assert set(f["flies"][0]) == {"id", "sex", "fly", "mode", "hz", "senses", "sps", "graded_eps", "driver"}
    assert set(f["world"]) == set(recording.WORLD_KEYS) and f["flies"][0]["fly"]["x"] is not None
    listed = recording.list_recordings()
    assert listed[0]["id"] == rid and listed[0]["ticks"] == 6 and listed[0]["physics"] is False and listed[0]["poses"] is False


def test_the_cap_refuses_a_start_and_names_the_folder(conn, rec_dir, monkeypatch):
    g = Game(build_brain(conn, "game", seed=0), seed=1)
    try:
        monkeypatch.setattr(recording, "SIZE_CAP", 10)
        rec_dir.mkdir(parents=True)
        (rec_dir / "big.bin").write_bytes(b"x" * 20)
        why = recording.cannot_start(g)
        assert why and "cap" in why and str(rec_dir) in why and "nothing is deleted" in why
        assert g.action({"type": "capture", "on": True}) == {"ok": False, "error": why}
        monkeypatch.setattr(recording, "SIZE_CAP", 2 * 1024 ** 3)
        (rec_dir / "big.bin").unlink()
        file_in_the_way = rec_dir / "file"
        file_in_the_way.write_text("x")
        monkeypatch.setattr(recording, "recordings_dir", lambda: file_in_the_way)
        assert "can't write" in recording.cannot_start(g)
    finally:
        g.close()


@pytest.fixture(scope="module")
def served(conn):
    game = Game(build_brain(conn, "game", seed=0), seed=1)
    server = ThreadingHTTPServer(("127.0.0.1", 0), S.make_handler(game))
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    try:
        yield game, f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
        game.close()


def _get(base, path, headers=None):
    req = urllib.request.Request(base + path, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, dict(r.headers), r.read()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read()


def _post(base, obj, headers=None):
    req = urllib.request.Request(base + "/api/action", data=json.dumps(obj).encode(), method="POST",
                                 headers={"Content-Type": "application/json", **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, dict(r.headers), json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), json.loads(e.read())


def test_the_replay_endpoints_send_no_wildcard_cors_header(served, rec_dir):
    game, base = served
    code, headers, body = _get(base, "/api/replays")
    assert code == 200 and json.loads(body) == {"ok": True, "replays": []}
    assert "Access-Control-Allow-Origin" not in headers
    code, headers, _ = _get(base, "/api/replay/20261003-120000/header")
    assert code == 404 and "Access-Control-Allow-Origin" not in headers
    assert _get(base, "/api/replay/../../etc/header")[0] in (400, 404)
    code, headers, _ = _get(base, "/api/state")
    assert code == 200 and headers["Access-Control-Allow-Origin"] == "*"          # the live endpoints as before


def test_only_the_servers_own_page_may_save_a_replay(served, rec_dir):
    game, base = served
    port = base.rsplit(":", 1)[1]
    code, headers, reply = _post(base, {"type": "capture", "on": True}, {"Origin": "http://evil.example"})
    assert code == 403 and reply["ok"] is False and "evil.example" in reply["error"] and "Access-Control-Allow-Origin" not in headers
    code, _, reply = _post(base, {"type": "capture", "on": True}, {"Host": "evil.example:80"})
    assert code == 403 and "Host" in reply["error"]
    code, _, reply = _post(base, {"type": "capture", "on": True}, {"Host": f"localhost:{int(port) + 1}"})
    assert code == 403
    assert game.capture is None
    for host, origin in ((f"127.0.0.1:{port}", None), (f"localhost:{port}", f"http://localhost:{port}"), (f"127.0.0.1:{port}", f"http://127.0.0.1:{port}")):
        hdr = {"Host": host, **({"Origin": origin} if origin else {})}
        code, _, reply = _post(base, {"type": "capture", "on": True}, hdr)
        assert code == 200 and reply == {"ok": True}, (hdr, reply)
        game.tick()                                     # the action lands on the next tick
        assert game.capture is not None
        code, _, reply = _post(base, {"type": "capture", "on": False}, hdr)
        assert code == 200 and reply["ok"]
        game.tick()
        assert game.capture is None
    # an action that does not write is not origin-checked (the live API is open to any page, as before)
    code, _, reply = _post(base, {"type": "pause", "on": False}, {"Origin": "http://evil.example"})
    assert code == 200 and reply["ok"]
    # and what was saved comes back through the endpoints, inflated by the client
    rid = recording.list_recordings()[0]["id"]
    code, headers, body = _get(base, f"/api/replay/{rid}/header")
    assert code == 200 and json.loads(body)["ticks"] == 1 and "Access-Control-Allow-Origin" not in headers
    code, headers, body = _get(base, f"/api/replay/{rid}/frames")
    assert code == 200 and headers["Content-Encoding"] == "gzip" and headers["Content-Type"] == "application/x-ndjson"
    assert len(gzip.decompress(body).splitlines()) == 1
    assert _get(base, f"/api/replay/{rid}/poses")[0] == 404              # a drawn run has no poses


@needs_flygym
def test_a_physics_pair_replay_reproduces_the_geoms_exactly(conn, rec_dir):
    import mujoco
    g = Game(build_brain(conn, "game", seed=0), seed=1, body="physics", brain_procs="off",
             partner={"conn": conn, "brain_kwargs": {"seed": 1000}, "parts": False})
    try:
        assert g.action({"type": "capture", "on": True})["ok"]
        for _ in range(9):
            g.tick()
        mujoco.mj_kinematics(g.pair_world._m, g.pair_world._d)        # after a step the positions are one step stale
        live = np.array(g.pair_world._d.geom_xpos)
        rid = g.capture.id
        assert g.action({"type": "capture", "on": False})["ok"]
        g.tick()
    finally:
        g.close()
    folder = recording.folder_of(rid)
    names = sorted(p.name for p in folder.iterdir())
    assert names == ["frames.jsonl.gz", "header.json", "model", "poses.f32", "qpos.npy", "t.npy"]
    h = recording.read_header(folder)
    assert h["physics"]["model"] == "two_flies.mjb.gz" and h["physics"]["fly_names"] == ["fly0", "fly1"] and h["physics"]["pairs"] == 233
    assert 5_000_000 < (folder / "model" / "two_flies.mjb.gz").stat().st_size < 40_000_000 and h["physics"]["engine"].startswith("flygym 2.1, MuJoCo 3.9")
    assert sorted(p.name for p in (folder / "model").iterdir()) == ["two_flies.mjb.gz"]      # the raw .mjb is not left behind
    assert h["poses"]["geoms"] == recording.atlas_geoms() and len(h["poses"]["geoms"]) == 69 and h.get("poses_error") is None
    q = np.load(folder / "qpos.npy", allow_pickle=False)
    t = np.load(folder / "t.npy", allow_pickle=False)
    assert q.shape == (9, h["physics"]["nq"]) and t.shape == (9,) and t[-1] == pytest.approx(0.225)
    m = recording.load_model(folder, h)
    d = mujoco.MjData(m)
    d.qpos[:] = q[-1]
    mujoco.mj_kinematics(m, d)
    assert np.abs(np.array(d.geom_xpos) - live).max() == 0.0             # the research's check: exact, not approximate
    poses = np.fromfile(folder / "poses.f32", dtype="<f4").reshape(9, 2, 69, 7)
    th = h["poses"]["geoms"].index("Thorax")
    assert np.ptp(poses[:, :, th, :], axis=(0, 1)).max() < 1e-5          # the thorax geom is fixed in its own body's frame
    assert np.allclose(np.linalg.norm(poses[..., 3:], axis=-1), 1.0, atol=1e-5)
    # the same convention as the 3-D view's gait atlas (models/nmf_gait.bin: poses in the thorax body's frame), so the
    # browser can feed a replay frame to the nodes the atlas feeds: the standing frame's thorax entry is this one
    from virtual_fly import connectome
    web = Path(connectome.PACKAGE_DIR) / "web" / "models"
    atlas_meta = json.loads((web / "nmf_gait.json").read_text())
    atlas = np.fromfile(web / "nmf_gait.bin", dtype="<f4").reshape(atlas_meta["frames"], 69, 7)
    standing = atlas[atlas_meta["standing"], th]
    assert np.abs(standing[:3] - poses[0, 0, th, :3]).max() < 1e-3
    assert min(np.abs(standing[3:] - poses[0, 0, th, 3:]).max(), np.abs(standing[3:] + poses[0, 0, th, 3:]).max()) < 1e-3
    assert recording.list_recordings()[0]["poses"] is True
