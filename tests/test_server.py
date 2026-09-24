"""The HTTP/SSE API (server.py) on a real ThreadingHTTPServer bound to a free port."""

import gzip
import http.client
import io
import json
import threading
import urllib.error
import urllib.parse
import urllib.request
from http.server import ThreadingHTTPServer

import numpy as np
import pytest

import virtual_fly.server as S
from virtual_fly.game import Game
from virtual_fly.settings import build_brain


@pytest.fixture(scope="module")
def served(conn, tmp_path_factory):
    """(game, base url): a game on the synthetic brain, ticked a few times, behind a live server.

    ``WEB_DIR`` is pointed at a temporary directory with a known index.html so the static-file
    branch is tested independently of what virtual_fly/web contains."""
    web = tmp_path_factory.mktemp("web")
    (web / "index.html").write_text("<!doctype html><title>synthetic fly</title>")
    (web / "app.js").write_text("console.log('fly');")
    old_web = S.WEB_DIR
    S.WEB_DIR = web
    game = Game(build_brain(conn, "game", seed=0), autopilot=False, seed=2)
    for _ in range(3):
        game.tick()
    server = ThreadingHTTPServer(("127.0.0.1", 0), S.make_handler(game))
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield game, f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
        S.WEB_DIR = old_web


def get(base, path, headers=None):
    req = urllib.request.Request(base + path, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, dict(r.headers), r.read()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read()


def get_json(base, path):
    code, headers, body = get(base, path)
    assert headers["Content-Type"] == "application/json", headers
    return code, json.loads(body)


def post(base, path, payload):
    data = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
    req = urllib.request.Request(base + path, data=data, method="POST", headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def test_layout_is_gzipped_when_accepted(served, conn):
    game, base = served
    code, headers, body = get(base, "/api/layout", {"Accept-Encoding": "gzip"})
    assert code == 200 and headers["Content-Encoding"] == "gzip" and int(headers["Content-Length"]) == len(body)
    assert headers["Cache-Control"] == "no-store" and headers["Access-Control-Allow-Origin"] == "*"
    layout = json.loads(gzip.decompress(body))
    assert layout["n"] == conn.n and len(body) < len(game.layout_json)
    code, headers, raw = get(base, "/api/layout")
    assert code == 200 and "Content-Encoding" not in headers and raw == game.layout_json


def test_state_endpoint(served):
    game, base = served
    code, state = get_json(base, "/api/state")
    assert code == 200 and state == game.state_dict and state["seq"] >= 3


def test_action_endpoint(served):
    game, base = served
    assert post(base, "/api/action", {"type": "zap", "spec": "MDN", "hz": 60}) == (200, {"ok": True, "n": 4})
    code, reply = post(base, "/api/action", {"type": "zap", "spec": "nothing"})
    assert code == 200 and reply["ok"] is False and "No neurons match" in reply["error"]
    code, reply = post(base, "/api/action", b"{not json")
    assert code == 200 and reply["ok"] is False and reply["error"]
    assert post(base, "/api/action", {"type": "scenario", "id": "nope"})[1]["ok"] is False
    assert not game.actions.empty()                                            # queued for the loop
    game.tick()
    assert game.actions.empty() and game.zaps and game.zaps[0][0] == "MDN"
    code, body = post(base, "/api/other", {})
    assert code == 404
    code, reply = post(base, "/api/action", [1, 2, 3])
    assert code == 200 and reply["ok"] is False and "JSON object" in reply["error"]


def test_post_to_an_unknown_path_drains_its_body(served):
    """A 404 must still consume the request body, or the next request on a keep-alive connection
    is parsed from the middle of the previous one."""
    game, base = served
    host, port = base[len("http://"):].split(":")
    c = http.client.HTTPConnection(host, int(port), timeout=10)
    body = json.dumps({"type": "zap", "spec": "MDN", "hz": 60, "padding": "x" * 5000}).encode()
    c.request("POST", "/api/other", body=body, headers={"Content-Type": "application/json"})
    resp = c.getresponse()
    assert resp.status == 404 and resp.read() == b"not found"
    c.request("GET", "/api/state")                                            # same connection
    resp = c.getresponse()
    assert resp.status == 200 and json.loads(resp.read())["seq"] >= 0
    c.request("POST", "/api/action", body=b'{"type": "pause", "on": false}', headers={"Content-Type": "application/json"})
    resp = c.getresponse()
    assert resp.status == 200 and json.loads(resp.read()) == {"ok": True}
    c.close()


def test_types_search(served):
    game, base = served
    assert get_json(base, "/api/types?q=LC10")[1] == {"ok": True, "types": [{"type": "LC10a", "n": 20}]}
    assert get_json(base, "/api/types")[1]["types"] == []
    assert len(get_json(base, "/api/types?q=DN&limit=3")[1]["types"]) == 3


def test_neuron_lookup(served, conn):
    game, base = served
    i = int(conn.select("MN9/L")[0])
    code, reply = get_json(base, f"/api/neuron?index={i}")
    n = reply["neuron"]
    assert code == 200 and n["index"] == i and n["type"] == "MN9" and n["side"] == "L"
    assert {"body_id", "superclass", "class", "subclass", "nt", "sign", "nerve", "neuromere", "hex", "soma",
            "n_inputs", "n_outputs", "rate_hz", "inputs", "outputs"} <= set(n)
    assert n["inputs"][0]["type"] == "GNG232" and n["outputs"] == [] and isinstance(n["rate_hz"], float)
    assert get_json(base, f"/api/neuron?body={conn.body_id[i]}")[1]["neuron"]["index"] == i
    assert get_json(base, "/api/neuron?index=999999")[0] == 404 and get_json(base, "/api/neuron?body=1")[0] == 404
    assert get_json(base, "/api/neuron")[0] == 404


def test_partners(served):
    game, base = served
    code, reply = get_json(base, "/api/partners?spec=MN9&dir=in&top=3")
    assert code == 200 and reply["spec"] == "MN9" and reply["n"] == 2 and reply["dir"] == "in" and len(reply["rows"]) == 3
    assert reply["rows"][0]["type"] == "GNG232" and "fraction" in reply["rows"][0]
    assert get_json(base, "/api/partners?spec=GNG232&dir=out")[1]["rows"][0]["type"] == "MN9"
    assert get_json(base, "/api/partners?spec=zzz")[0] == 400 and get_json(base, "/api/partners")[0] == 400


def test_trace(served, conn):
    game, base = served
    q = urllib.parse.urlencode({"from": "LC10a/L", "to": "DNa02/L", "hops": 9, "top": 1})
    code, reply = get_json(base, f"/api/trace?{q}")
    assert code == 200 and reply["from"] == "LC10a/L" and reply["secs"] >= 0
    assert [p["nodes"] for p in reply["paths"]] == [["LC10a/L", "AOTU019/L", "DNa02/L"]]
    assert reply["relays"] == [["AOTU019/L", pytest.approx(reply["paths"][0]["score"])]]
    pts = reply["neurons"][0]
    assert len(pts) == 3 and conn.types[pts[1]] == "AOTU019" and game.has_soma[pts[1]]
    assert get_json(base, "/api/trace?" + urllib.parse.urlencode({"from": "LC10a/L", "to": "DNa02/L", "avoid": "AOTU019"}))[1]["paths"][0]["nodes"][1] == "AOTU025/L"
    assert get_json(base, "/api/trace?from=LC10a/L&to=nothing")[0] == 400


def test_history_learning_decoder(served):
    game, base = served
    code, reply = get_json(base, "/api/history?keys=MN9,GF,nope&n=2")
    assert code == 200 and reply["bin_ms"] == 25.0 and set(reply["history"]) == {"MN9", "GF"}
    assert len(reply["history"]["MN9"]) == 2
    assert len(get_json(base, "/api/history?keys=MN9")[1]["history"]["MN9"]) == len(game.brain.monitors["MN9"].history) >= 3
    assert set(get_json(base, "/api/history")[1]["history"]) == set(game.brain.monitors)
    code, reply = get_json(base, "/api/learning")
    assert reply["learning"]["MBON11"]["valence"] == 1 and reply["settings"]["plastic_synapses"] == 240
    assert reply["depressed_fraction"] == 0.0
    assert "DNp01" in get_json(base, "/api/decoder")[1]["targets"]


def test_recording_and_spikes_downloads(served, conn):
    game, base = served
    code, headers, body = get(base, "/api/recording")
    assert code == 200 and headers["Content-Disposition"].endswith("fly-session.json")
    assert json.loads(body)["frames"] == [] and json.loads(body)["settings"]["dt"] == 0.5
    game.recording = [{"t": 1.0}]
    assert json.loads(get(base, "/api/recording")[2])["frames"] == [{"t": 1.0}]
    game.recording = None
    game.brain.recording = [(4, np.array([1, 2], dtype=np.int32))]
    code, headers, body = get(base, "/api/spikes", {"Accept-Encoding": "gzip"})
    assert code == 200 and headers["Content-Type"] == "application/octet-stream"
    if headers.get("Content-Encoding") == "gzip":
        body = gzip.decompress(body)
    z = np.load(io.BytesIO(body))
    assert z["time_ms"].tolist() == [2.0, 2.0] and z["neuron"].tolist() == [1, 2]
    assert z["body_id"].tolist() == conn.body_id[[1, 2]].tolist()
    game.brain.recording = None
    assert np.load(io.BytesIO(get(base, "/api/spikes")[2]))["neuron"].size == 0


def test_stream_sends_the_current_state_first(served):
    game, base = served
    host, port = base[len("http://"):].split(":")
    c = http.client.HTTPConnection(host, int(port), timeout=10)
    c.request("GET", "/api/stream")
    resp = c.getresponse()
    assert resp.status == 200 and resp.getheader("Content-Type") == "text/event-stream"
    assert resp.readline() == b"retry: 1000\n"
    line = resp.readline()
    while not line.startswith(b"data: "):
        line = resp.readline()
    first = json.loads(line[len(b"data: "):])
    assert first == game.state_dict and len(game.subscribers) == 1
    game.tick()                                                                # published to the subscriber
    assert resp.readline() == b"\n"
    line = resp.readline()
    assert line.startswith(b"data: ") and json.loads(line[6:])["seq"] == game.state_dict["seq"]
    c.close()


def test_genes_lines_and_driver_endpoints(served, conn, tmp_path):
    from virtual_fly import genetics as G
    from tests.test_genetics import make_fake
    game, base = served
    code, g = get_json(base, "/api/genes")
    assert code == 200 and g["ok"] and {e["key"] for e in g["expression"]} == {"fru", "dsx", "both", "male", "dimorphic"}
    assert g["readouts"]["pIP10"]["tags"] == ["fru", "♂"] and any(t["nt"] == "gaba" and t["sign"] == -1 for t in g["transmitters"])
    fetch, calls, body, other = make_fake(conn)
    game.neuronbridge = G.NeuronBridge(cache_dir=tmp_path, fetch=fetch)
    code, r = get_json(base, "/api/lines?spec=pIP10/L")
    assert code == 200 and r["ok"] and [l["line"] for l in r["lines"]] == ["SS00001", "R00A00"] and r["sampled"] == [body]
    code, r = get_json(base, "/api/driver?line=SS00001")
    assert code == 200 and r["ok"] and r["neurons"][0]["type"] == "pIP10" and r["spec"].startswith(f"body:{body}")
    assert get_json(base, "/api/lines")[0] == 400 and get_json(base, "/api/driver")[0] == 400
    assert get_json(base, "/api/lines?spec=NOPE")[0] == 400 and get_json(base, "/api/driver?line=SS99999")[0] == 400
    game.neuronbridge = G.NeuronBridge(cache_dir=tmp_path / "off", fetch=lambda url, timeout: (_ for _ in ()).throw(OSError("down")))
    code, r = get_json(base, "/api/lines?spec=pIP10")
    assert code == 502 and "internet" in r["error"]
    code, r = get_json(base, f"/api/neuron?index={int(conn.select('pIP10/L')[0])}")
    assert code == 200 and [x["symbol"] for x in r["neuron"]["genes"]] == ["fru", "ChAT", "VAChT"]


def test_genome_endpoint(served):
    game, base = served
    code, r = get_json(base, "/api/genome")
    assert code == 200 and r["ok"] and r["level"] == "real" and [l["level"] for l in r["levels"]][0] == "real"


def test_static_files_and_404s(served):
    game, base = served
    code, headers, body = get(base, "/")
    assert code == 200 and headers["Content-Type"].startswith("text/html") and b"synthetic fly" in body
    assert get(base, "/index.html")[2] == body
    code, headers, body = get(base, "/app.js")
    assert code == 200 and "javascript" in headers["Content-Type"] and b"console" in body
    assert get(base, "/missing.js")[0] == 404 and get(base, "/api/nothing")[0] == 404
    assert get(base, "/../connectome.py")[0] == 404                             # no path traversal
    assert get(base, "/../../tests/conftest.py")[0] == 404
    assert get(base, "//etc/passwd")[0] == 404
    code, headers, body = get(base, "/api/nothing")
    assert body == b"not found" and headers["Content-Type"] == "text/plain"


def test_options_preflight(served):
    game, base = served
    host, port = base[len("http://"):].split(":")
    c = http.client.HTTPConnection(host, int(port), timeout=10)
    c.request("OPTIONS", "/api/action")
    resp = c.getresponse()
    assert resp.status == 204 and resp.getheader("Access-Control-Allow-Methods") == "GET, POST, OPTIONS"
    c.close()


def test_serve_scans_ports_and_opens_the_browser(monkeypatch, capsys):
    opened, made = [], []

    class FakeServer:
        def __init__(self, addr, handler):
            if addr[1] == 9100:
                raise OSError("port busy")
            self.server_address = addr
            made.append(addr)

        def serve_forever(self):
            raise KeyboardInterrupt

    class FakeGame:
        conn = None

        def loop(self):
            pass

    monkeypatch.setattr(S, "ThreadingHTTPServer", FakeServer)
    monkeypatch.setattr(S.webbrowser, "open", lambda url: opened.append(url))
    server = S.serve(FakeGame(), port=9100, open_browser=True, host="127.0.0.1")
    assert made == [("127.0.0.1", 9101)] and opened == ["http://127.0.0.1:9101/"] and server.daemon_threads
    assert "alive at http://127.0.0.1:9101/" in capsys.readouterr().out
    monkeypatch.setattr(S, "ThreadingHTTPServer", lambda *a: (_ for _ in ()).throw(OSError("busy")))
    with pytest.raises(SystemExit, match="free port"):
        S.serve(FakeGame(), port=9100, open_browser=False)
