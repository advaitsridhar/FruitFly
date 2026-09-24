"""
A small web server: the browser draws everything and sends your mouse and clicks back.

Only the standard library is used. The game state is pushed to the page over a Server-Sent
Events stream (``/api/stream``); the page falls back to polling ``/api/state`` if the stream
breaks. Everything else is a plain JSON API, so any program that can make HTTP requests (a game
engine, a notebook, a robot) can drive the fly:

    GET  /api/layout                  static data: brain map, arena, readouts, presets, odours ...
    GET  /api/state                   the latest game state (also streamed on /api/stream)
    POST /api/action  {"type": ...}   see Game.action; e.g. {"type": "zap", "spec": "MDN", "hz": 60}
    GET  /api/types?q=LC10            search cell types
    GET  /api/neuron?index=123        everything known about one neuron (or ?body=<bodyId>)
    GET  /api/partners?spec=MN9&dir=in    strongest input (or output) types of a population
    GET  /api/trace?from=LC10a/L&to=DNa02/L&hops=4   strongest wiring routes
    GET  /api/history?keys=MN9,GF     rate histories of readouts (one value per tick)
    GET  /api/learning                per-MBON synaptic strengths and dopamine
    GET  /api/recording               the recorded session (JSON), if recording
    GET  /api/spikes                  the recorded spikes (npz) when recording with spikes
"""

from __future__ import annotations

import gzip
import io
import json
import mimetypes
import queue
import threading
import time
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

from .pathways import relay_ranking, strongest_partners, trace

WEB_DIR = Path(__file__).resolve().parent / "web"


def make_handler(game):
    conn = game.conn

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def _send(self, code, body, ctype, extra=None):
            if len(body) > 4096 and "gzip" in (self.headers.get("Accept-Encoding") or ""):
                body = gzip.compress(body, 3)
                extra = {**(extra or {}), "Content-Encoding": "gzip"}
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Access-Control-Allow-Origin", "*")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code=200):
            self._send(code, json.dumps(obj, separators=(",", ":")).encode(), "application/json")

        def _error(self, msg, code=400):
            self._json({"ok": False, "error": str(msg)}, code)

        def do_OPTIONS(self):
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.end_headers()

        def do_GET(self):
            url = urllib.parse.urlparse(self.path)
            path, q = url.path, urllib.parse.parse_qs(url.query)
            get = lambda k, d=None: q.get(k, [d])[0]
            try:
                if path in ("/", "/index.html"):
                    return self._send(200, (WEB_DIR / "index.html").read_bytes(), "text/html; charset=utf-8")
                if path == "/api/state":
                    return self._send(200, game.state_json, "application/json")
                if path == "/api/layout":
                    return self._send(200, game.layout_json, "application/json")
                if path == "/api/stream":
                    return self._stream()
                if path == "/api/types":
                    text = get("q", "")
                    hits = conn.find_types(text, limit=int(get("limit", 50))) if text else []
                    return self._json({"ok": True, "types": [{"type": t, "n": n} for t, n in hits]})
                if path == "/api/neuron":
                    if get("body"):
                        idx = conn.select(f"body:{int(get('body'))}")
                        if idx.size == 0:
                            return self._error("no such bodyId", 404)
                        i = int(idx[0])
                    else:
                        i = int(get("index", -1))
                        if not 0 <= i < conn.n:
                            return self._error("index out of range", 404)
                    info = conn.info(i)
                    info["rate_hz"] = float(game.brain.spike_count[i]) / max(game.brain.window_ms, 1) * 1000.0
                    info["inputs"] = conn.inputs_of(f"index:{i}", top=8)
                    info["outputs"] = conn.outputs_of(f"index:{i}", top=8)
                    return self._json({"ok": True, "neuron": info})
                if path == "/api/partners":
                    spec = get("spec", "")
                    if not spec or conn.count(spec) == 0:
                        return self._error("no neurons match that spec")
                    direction = "in" if get("dir", "in") == "in" else "out"
                    rows = strongest_partners(conn, spec, direction, top=int(get("top", 15)))
                    return self._json({"ok": True, "spec": spec, "n": conn.count(spec), "dir": direction, "rows": rows})
                if path == "/api/trace":
                    src, dst = get("from", ""), get("to", "")
                    if conn.count(src) == 0 or conn.count(dst) == 0:
                        return self._error("no neurons match one of the specs")
                    t0 = time.time()
                    paths = trace(conn, src, dst, max_hops=min(6, int(get("hops", 4))), top=int(get("top", 8)),
                                  avoid=get("avoid") or None)
                    soma_paths = []
                    tg = conn.type_graph()
                    for p in paths:
                        pts = []
                        for name in p.nodes:
                            node = tg.index.get(name)
                            members = np.flatnonzero(tg.node_of_neuron == node) if node is not None else []
                            members = [int(m) for m in members if game.has_soma[m]][:1]
                            pts.append(members[0] if members else None)
                        soma_paths.append(pts)
                    return self._json({"ok": True, "from": src, "to": dst, "secs": round(time.time() - t0, 2),
                                       "paths": [p.to_dict() for p in paths], "neurons": soma_paths,
                                       "relays": relay_ranking(paths)[:10]})
                if path == "/api/history":
                    keys = [k for k in get("keys", "").split(",") if k]
                    n = int(get("n", 400))
                    out = {}
                    for k in keys or list(game.brain.monitors):
                        m = game.brain.monitors.get(k)
                        if m is not None:
                            out[k] = m.history[-n:]
                    return self._json({"ok": True, "bin_ms": game.brain.monitors and next(iter(game.brain.monitors.values())).bin_ms,
                                       "history": out})
                if path == "/api/learning":
                    pl = game.brain.plasticity
                    if pl is None:
                        return self._json({"ok": True, "learning": None})
                    return self._json({"ok": True, "learning": pl.summary(conn), "settings": pl.settings(),
                                       "depressed_fraction": pl.depressed_fraction()})
                if path == "/api/decoder":
                    return self._json({"ok": True, "targets": game.decoder.dn_targets})
                if path == "/api/recording":
                    rec = game.recording or []
                    body = json.dumps({"frames": rec, "settings": game.brain.settings()}).encode()
                    return self._send(200, body, "application/json",
                                      {"Content-Disposition": "attachment; filename=fly-session.json"})
                if path == "/api/spikes":
                    b = game.brain
                    rec = b.recording or []
                    t_ms, idx = b.recording_arrays(rec)
                    buf = io.BytesIO()
                    np.savez_compressed(buf, time_ms=t_ms, neuron=idx, body_id=conn.body_id[idx] if idx.size else idx)
                    return self._send(200, buf.getvalue(), "application/octet-stream",
                                      {"Content-Disposition": "attachment; filename=fly-spikes.npz"})
                # static files
                rel = path.lstrip("/")
                target = (WEB_DIR / rel).resolve()
                if rel and target.is_file() and WEB_DIR in target.parents:
                    ctype = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
                    return self._send(200, target.read_bytes(), ctype)
                return self._send(404, b"not found", "text/plain")
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as e:
                import traceback
                traceback.print_exc()
                try:
                    self._error(e, 500)
                except Exception:
                    pass

        def _stream(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "keep-alive")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            q = game.subscribe()
            try:
                self.wfile.write(b"retry: 1000\n\n")
                self.wfile.write(b"data: " + game.state_json + b"\n\n")
                self.wfile.flush()
                while True:
                    try:
                        payload = q.get(timeout=2.0)
                    except queue.Empty:
                        self.wfile.write(b": keepalive\n\n")
                        self.wfile.flush()
                        continue
                    # drop stale frames if the client is slow
                    while not q.empty():
                        payload = q.get_nowait()
                    self.wfile.write(b"data: " + payload + b"\n\n")
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass
            finally:
                game.unsubscribe(q)

        def do_POST(self):
            if self.path != "/api/action":
                return self._send(404, b"not found", "text/plain")
            try:
                length = int(self.headers.get("Content-Length") or 0)
                data = json.loads(self.rfile.read(length) or b"{}")
                reply = game.action(data)
            except Exception as e:
                reply = {"ok": False, "error": str(e)}
            self._json(reply)

    return Handler


def serve(game, port: int = 8765, open_browser: bool = True, host: str = "127.0.0.1"):
    server = None
    for p in range(port, port + 20):
        try:
            server = ThreadingHTTPServer((host, p), make_handler(game))
            break
        except OSError:
            continue
    if server is None:
        raise SystemExit("Could not find a free port. Try: python fly_game.py --port 9000")
    server.daemon_threads = True
    threading.Thread(target=game.loop, daemon=True).start()
    url = f"http://{host}:{server.server_address[1]}/"
    print(f"\nThe fly is alive at {url}\n(keep this window open; press Ctrl+C here to quit)\n")
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Bye!")
    return server
