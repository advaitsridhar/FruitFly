"""The page in a real browser (Playwright, headless Chromium): the 3-D dish of docs/TWO_FLIES_PLAN.md 7.6.

Skipped unless ``playwright`` imports, its Chromium is installed and ``VF_BROWSER_TESTS=1`` is set, so CI (which installs
only the ``dev`` extra) never runs it. Locally: ``pip install -e ".[browser]"``, then ``python -m playwright install chromium``
(on Linux ``python -m playwright install-deps`` too, with sudo), then ``VF_BROWSER_TESTS=1 python -m pytest -q tests/test_browser.py``.

The server runs on the synthetic connectome with the real ``virtual_fly/web`` folder (so the vendored three.js and the model
files are the ones a user gets). Headless Chromium may draw WebGL with a software renderer, so frame rates are reported
(printed), never asserted.
"""
import json
import os
import threading
import time
from http.server import ThreadingHTTPServer

import pytest

import virtual_fly.server as S
from virtual_fly.game import Game
from virtual_fly.settings import build_brain

playwright = pytest.importorskip("playwright", reason="the browser tests need the playwright package (the `browser` extra)")
from playwright.sync_api import sync_playwright  # noqa: E402

pytestmark = pytest.mark.skipif(os.environ.get("VF_BROWSER_TESTS") != "1",
                                reason="browser tests run only with VF_BROWSER_TESTS=1 (they need Playwright's Chromium)")

CHROMIUM_ARGS = ["--enable-unsafe-swiftshader", "--ignore-gpu-blocklist", "--use-gl=angle", "--use-angle=swiftshader"]


def _chromium_available():
    try:
        with sync_playwright() as p:
            b = p.chromium.launch(headless=True, args=CHROMIUM_ARGS)
            b.close()
        return True
    except Exception:
        return False


@pytest.fixture(scope="module")
def chromium_ok():
    if not _chromium_available():
        pytest.skip("Playwright's Chromium is not installed (python -m playwright install chromium)")
    return True


def _serve(game):
    server = ThreadingHTTPServer(("127.0.0.1", 0), S.make_handler(game))
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"


@pytest.fixture(scope="module")
def one_fly(conn, mini_vfb, chromium_ok):
    """A single synthetic fly behind a live server, its loop running so the page gets states."""
    game = Game(build_brain(conn, "game", seed=0), seed=2)
    threading.Thread(target=game.loop, daemon=True).start()
    server, base = _serve(game)
    try:
        yield game, base
    finally:
        game.close()
        server.shutdown(); server.server_close()


@pytest.fixture(scope="module")
def two_flies(conn, mini_vfb, chromium_ok):
    """The synthetic male with a synthetic female partner (tests/synthetic_connectome.py), brains in this process."""
    from synthetic_connectome import build_synthetic
    from virtual_fly.connectome import load_connectome
    import tempfile, pathlib
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="vf-browser-"))
    fpath = tmp / "synthetic-female.flyb.gz"
    build_synthetic(fpath, sex="female")
    fconn = load_connectome(str(fpath), quiet=True)
    game = Game(build_brain(conn, "game", seed=0), seed=3, brain_procs="off", partner={"conn": fconn})
    threading.Thread(target=game.loop, daemon=True).start()
    server, base = _serve(game)
    try:
        yield game, base
    finally:
        game.close()
        server.shutdown(); server.server_close()


class _Page:
    """A Chromium page on the game, with the requests and the console errors it made."""

    def __init__(self, p, base, init_script=None, query=""):
        self.browser = p.chromium.launch(headless=True, args=CHROMIUM_ARGS)
        self.page = self.browser.new_page(viewport={"width": 1280, "height": 800})
        if init_script:
            self.page.add_init_script(init_script)
        self.requests, self.errors = [], []
        self.page.on("request", lambda r: self.requests.append(r.url))
        self.page.on("console", lambda m: self.errors.append(m.text) if m.type == "error" else None)
        self.page.on("pageerror", lambda e: self.errors.append(str(e)))
        self.page.goto(base + "/" + query, wait_until="load")
        self.page.wait_for_function("() => document.getElementById('loading').hidden === true", timeout=30000)

    def close(self):
        self.browser.close()

    def toggle_3d(self):
        self.page.click("#view3dBtn")

    def wait_3d(self, timeout=60000):
        self.page.wait_for_function("() => window.__vf3d && window.__vf3d.on === true && window.__vf3d.flies > 0", timeout=timeout)
        return self.page.evaluate("() => window.__vf3d")


BLOCK_WEBGL2 = """
(() => { const orig = HTMLCanvasElement.prototype.getContext;
  HTMLCanvasElement.prototype.getContext = function (type, ...rest) { if (type === "webgl2") return null; return orig.call(this, type, ...rest); }; })();
"""


def test_the_default_page_fetches_nothing_of_the_3d_view(one_fly):
    game, base = one_fly
    with sync_playwright() as p:
        pg = _Page(p, base)
        try:
            time.sleep(1.5)
            fetched = [u for u in pg.requests if "/vendor/three/" in u or "/models/" in u or "arena3d.js" in u]
            assert fetched == [], fetched
            assert pg.page.evaluate("() => document.getElementById('arena3d').hidden") is True
            assert pg.errors == [], pg.errors
        finally:
            pg.close()


def test_the_3d_view_shows_the_fly_with_webgl2_and_its_badge(one_fly):
    game, base = one_fly
    with sync_playwright() as p:
        pg = _Page(p, base)
        try:
            pg.toggle_3d()
            info = pg.wait_3d()
            assert info["flies"] == 1 and info["triangles"] > 0 and info["lost"] is False and info["missingNodes"] == 0
            assert pg.page.evaluate("() => document.getElementById('arena3d').hidden") is False
            assert pg.page.evaluate("() => document.getElementById('arena').hidden") is True
            assert pg.page.is_visible("#badge3d") and "not physics" in pg.page.inner_text("#badge3d")
            vendor = [u for u in pg.requests if "/vendor/three/" in u]
            models = [u for u in pg.requests if "/models/" in u]
            assert any(u.endswith("three.module.js") for u in vendor) and any(u.endswith("nmf_fly.glb") for u in models)
            # a WebGL2 context on the 3-D canvas
            assert pg.page.evaluate("() => !!document.getElementById('arena3d').getContext('webgl2')") is True
            assert [e for e in pg.errors if "favicon" not in e] == [], pg.errors
            # the camera presets and the toggle back
            pg.page.select_option("#cam3d", "follow")
            time.sleep(0.5)
            assert pg.page.evaluate("() => window.__vf3d.camera") == "follow"
            pg.toggle_3d()
            assert pg.page.evaluate("() => document.getElementById('arena').hidden") is False
        finally:
            pg.close()


def test_without_webgl2_the_page_stays_in_2d_and_says_why(one_fly):
    game, base = one_fly
    with sync_playwright() as p:
        pg = _Page(p, base, init_script=BLOCK_WEBGL2)
        try:
            pg.toggle_3d()
            pg.page.wait_for_function("() => document.getElementById('view3dBtn').title.includes('WebGL2')", timeout=30000)
            assert pg.page.evaluate("() => document.getElementById('arena3d').hidden") is True
            assert pg.page.evaluate("() => document.getElementById('arena').hidden") is False
            assert "stays in 2-D" in pg.page.inner_text("#toast") or "stays in 2-D" in pg.page.evaluate("() => document.getElementById('view3dBtn').title")
            assert not any("/vendor/three/" in u for u in pg.requests), "three.js must not be fetched without WebGL2"
        finally:
            pg.close()


def test_with_a_partner_both_flies_are_in_the_scene(two_flies):
    game, base = two_flies
    with sync_playwright() as p:
        pg = _Page(p, base)
        try:
            pg.page.wait_for_function("() => document.getElementById('focusSel').hidden === false", timeout=30000)
            pg.toggle_3d()
            info = pg.wait_3d()
            assert info["flies"] == 2 and info["missingNodes"] == 0
            assert [e for e in pg.errors if "favicon" not in e] == [], pg.errors
        finally:
            pg.close()


def test_frame_rate_probe_reports(one_fly):
    """Reported, never asserted: headless Chromium draws with a software renderer (plan 7.5)."""
    game, base = one_fly
    with sync_playwright() as p:
        pg = _Page(p, base)
        try:
            pg.toggle_3d()
            pg.wait_3d()
            time.sleep(3.5)
            info = pg.page.evaluate("() => window.__vf3d")
            gl = pg.page.evaluate("() => { const c = document.createElement('canvas'); const g = c.getContext('webgl2'); const d = g && g.getExtension('WEBGL_debug_renderer_info'); return d ? g.getParameter(d.UNMASKED_RENDERER_WEBGL) : (g ? 'webgl2' : 'none'); }")
            print(f"\n3-D view in headless Chromium: {info['fps']} fps, {info['triangles']:,} triangles, {info['calls']} draw calls, renderer {gl!r} "
                  f"(headless: a software renderer is likely; the plan's 50 fps target is for a headed browser on the GPU)")
            assert info["fps"] >= 0
        finally:
            pg.close()


# ---------------------------------------------------------------- record-and-replay (docs/TWO_FLIES_PLAN.md 8.7-8.8)
def _save_replay(game, ticks=12):
    """A short recording of the served game (its loop is running): the capture action on, at least ``ticks`` ticks, off."""
    from virtual_fly import recording
    r = game.action({"type": "capture", "on": True})
    assert r["ok"], r
    deadline = time.time() + 15
    while time.time() < deadline:
        c = game.state_dict.get("capture")
        if c and c["ticks"] >= ticks:
            break
        time.sleep(0.05)
    r = game.action({"type": "capture", "on": False})
    assert r["ok"], r
    while game.capture is not None and time.time() < deadline:
        time.sleep(0.05)
    assert game.capture is None
    return recording.list_recordings()[0]["id"]


def test_a_saved_replay_plays_in_the_page_and_leaves_cleanly(one_fly, tmp_path, monkeypatch):
    from virtual_fly import recording
    monkeypatch.setattr(recording, "recordings_dir", lambda: tmp_path / "recordings")
    game, base = one_fly
    rid = _save_replay(game)
    with sync_playwright() as p:
        pg = _Page(p, base)
        try:
            pg.page.click("#replayRefresh")
            pg.page.wait_for_function(f"() => document.querySelector('#replayList button[data-id=\"{rid}\"]') !== null", timeout=10000)
            assert pg.page.evaluate("() => document.getElementById('player').hidden") is True
            pg.page.click(f'#replayList button[data-id="{rid}"]')
            pg.page.wait_for_function("() => window.__vfReplay().active && window.__vfReplay().frame >= 2", timeout=20000)
            assert pg.page.is_visible("#player") and pg.page.is_visible("#replayBadge")
            assert "replay of a recorded run" in pg.page.inner_text("#replayBadge")
            info = pg.page.evaluate("() => window.__vfReplay()")
            assert info["id"] == rid and info["frames"] >= 12 and info["poses"] is False
            assert pg.page.inner_text("#stRtf").startswith("replay")
            # the time advances while it plays, and the live ticks keep arriving behind it without being shown
            pg.page.wait_for_function("() => !window.__vfReplay().playing", timeout=20000)          # 12 ticks at 1x: 0.3 s
            a = pg.page.evaluate("() => window.__vfReplay()")
            time.sleep(0.4)
            b = pg.page.evaluate("() => window.__vfReplay()")
            # (a recording keeps the live game's own tick numbers, so the shown one stays put and below the live one)
            assert b["liveSeq"] > a["liveSeq"] and b["shownSeq"] == a["shownSeq"] and b["shownSeq"] < a["liveSeq"]
            # scrubbing: to the start, then to the end, where the time reads the whole length
            pg.page.evaluate("() => { const s = document.getElementById('plScrub'); s.value = '0'; s.dispatchEvent(new Event('input')); }")
            pg.page.wait_for_function("() => window.__vfReplay().frame === 0", timeout=5000)
            pg.page.evaluate("() => { const s = document.getElementById('plScrub'); s.value = s.max; s.dispatchEvent(new Event('input')); }")
            pg.page.wait_for_function("() => window.__vfReplay().frame === window.__vfReplay().frames - 1", timeout=5000)
            now, total = pg.page.inner_text("#plTime").split(" / ")
            assert now + " s" == total and total != "– s"
            # play again from the end starts over
            pg.page.click("#plPlay")
            pg.page.wait_for_function("() => window.__vfReplay().playing && window.__vfReplay().frame < 3", timeout=5000)
            # Leave: the player goes, the live fly is back and its ticks show again
            pg.page.click("#plLeave")
            pg.page.wait_for_function("() => !window.__vfReplay().active && document.getElementById('player').hidden", timeout=5000)
            assert pg.page.evaluate("() => document.getElementById('replayBadge').hidden") is True
            c = pg.page.evaluate("() => window.__vfReplay()")
            pg.page.wait_for_function(f"() => window.__vfReplay().shownSeq > {c['shownSeq']} + 5", timeout=10000)
            assert not pg.page.inner_text("#stRtf").startswith("replay")
            assert [e for e in pg.errors if "favicon" not in e] == [], pg.errors
        finally:
            pg.close()


def _physics_available():
    from virtual_fly import physics
    return physics.available()


@pytest.mark.skipif(not _physics_available(), reason="the physics pair needs flygym (optional)")
def test_a_physics_pair_replay_moves_the_legs_from_the_recording(one_fly, conn, tmp_path, monkeypatch):
    """A recording of two physics bodies carries every geom's pose per tick: the 3-D view moves the legs from it (the
    recorded physics), not from the gait animation, and says so on its badge."""
    from virtual_fly import recording
    monkeypatch.setattr(recording, "recordings_dir", lambda: tmp_path / "recordings")
    game, base = one_fly
    g = Game(build_brain(conn, "game", seed=0), seed=1, body="physics", brain_procs="off",
             partner={"conn": conn, "brain_kwargs": {"seed": 1000}, "parts": False})
    try:
        assert g.action({"type": "capture", "on": True})["ok"]
        for _ in range(6):
            g.tick()
        rid = g.capture.id
        assert g.action({"type": "capture", "on": False})["ok"]
        g.tick()
    finally:
        g.close()
    assert recording.list_recordings()[0]["poses"] is True
    with sync_playwright() as p:
        pg = _Page(p, base, query=f"?replay={rid}")
        try:
            pg.page.wait_for_function("() => window.__vfReplay().active && window.__vfReplay().poses === true", timeout=30000)
            pg.toggle_3d()
            info = pg.wait_3d()
            assert info["flies"] == 2 and info["missingNodes"] == 0
            pg.page.wait_for_function("() => window.__vf3d.replayPoses === true", timeout=20000)
            assert "physics" in pg.page.inner_text("#badge3dText") and "replay" in pg.page.inner_text("#badge3dText")
            assert pg.page.is_visible("#replayBadge")
            pg.page.click("#plLeave")
            pg.page.wait_for_function("() => !window.__vfReplay().active", timeout=5000)
            time.sleep(0.5)
            assert pg.page.evaluate("() => window.__vf3d.replayPoses") is False
            assert "not physics" in pg.page.inner_text("#badge3dText")
            assert [e for e in pg.errors if "favicon" not in e] == [], pg.errors
        finally:
            pg.close()
