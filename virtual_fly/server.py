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
    GET  /api/ontology?q=lobula       anatomy-ontology classes matching a text (or ?id=FBbt_00003870 for one class)
    GET  /api/partners?spec=MN9&dir=in    strongest input (or output) types of a population
    GET  /api/trace?from=LC10a/L&to=DNa02/L&hops=4   strongest wiring routes
    GET  /api/history?keys=MN9,GF     rate histories of readouts (one value per tick)
    GET  /api/learning                per-MBON synaptic strengths and dopamine
    GET  /api/genes                   the gene-expression populations, transmitter groups, FlyBase links
    GET  /api/genome                  the genome levels and the current fly's growth / survival status
    GET  /api/parts                   the parts list: modulators, graded cell types, counts, the tones right now
    GET  /api/lines?spec=pIP10        driver lines matching a population (NeuronBridge; needs internet)
    GET  /api/driver?line=SS02385     MaleCNS neurons a driver line labels (NeuronBridge; needs internet)
    GET  /api/recording               the recorded session (JSON), if recording
    GET  /api/spikes                  the recorded spikes (npz) when recording with spikes
"""

from __future__ import annotations

import errno
import gzip
import io
import json
import mimetypes
import queue
import re
import socket
import threading
import time
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

from . import genetics, recording, vfb, wiring
from . import parts as partslib

from .pathways import relay_ranking, strongest_partners, trace

# module scripts and glTF binaries get their proper types whatever the machine's registry says: a Windows registry can make
# .js come back as text/plain, and browsers refuse module scripts served that way (docs/TWO_FLIES_PLAN.md 7.1)
mimetypes.add_type("text/javascript", ".js")
mimetypes.add_type("text/javascript", ".mjs")
mimetypes.add_type("model/gltf-binary", ".glb")

WEB_DIR = Path(__file__).resolve().parent / "web"


def make_handler(game):

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def _send(self, code, body, ctype, extra=None, cors=True, compress=True):
            """``cors``: the wildcard CORS header every live endpoint carries; the replay endpoints send none (8.8).
            ``compress``: gzip a big body for a client that takes it (not a body that is gzip already)."""
            if compress and len(body) > 4096 and "gzip" in (self.headers.get("Accept-Encoding") or ""):
                body = gzip.compress(body, 3)
                extra = {**(extra or {}), "Content-Encoding": "gzip"}
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            if cors:
                self.send_header("Access-Control-Allow-Origin", "*")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code=200, cors=True):
            self._send(code, json.dumps(obj, separators=(",", ":")).encode(), "application/json", cors=cors)

        def _error(self, msg, code=400, cors=True):
            self._json({"ok": False, "error": str(msg)}, code, cors=cors)

        def _own_page(self):
            """None when the request comes from this server's own page (no Origin header, or ours, and a Host of
            localhost or 127.0.0.1 with our port), else why not: the one action that writes to disk takes nothing else,
            so another page open in the owner's browser cannot fill the disk (docs/TWO_FLIES_PLAN.md 8.8)."""
            port = self.server.server_address[1]
            host = (self.headers.get("Host") or "").strip()
            name, _, hport = host.partition(":")
            if name not in ("localhost", "127.0.0.1") or (hport and hport != str(port)):
                return f"refused: the Host header is {host!r}, not this server's localhost:{port} or 127.0.0.1:{port}"
            origin = self.headers.get("Origin")
            if origin is not None and origin.rstrip("/") not in (f"http://localhost:{port}", f"http://127.0.0.1:{port}"):
                return f"refused: the request comes from {origin!r}, not from this server's own page"
            return None

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
                if path == "/api/stream":
                    return self._stream()
                # the replays saved to disk (recording.py): no wildcard CORS header on any of these (8.8)
                if path == "/api/replays":
                    return self._json({"ok": True, "replays": recording.list_recordings()}, cors=False)
                m = re.fullmatch(r"/api/replay/([^/]{1,40})/(header|frames|poses)", path)
                if m:
                    folder = recording.folder_of(m.group(1))
                    if folder is None:
                        return self._error("no such replay", 404, cors=False)
                    what = m.group(2)
                    if what == "header":
                        return self._send(200, (folder / "header.json").read_bytes(), "application/json", cors=False)
                    if what == "frames":                # stored gzipped: sent as it is, the browser's fetch inflates it
                        return self._send(200, (folder / "frames.jsonl.gz").read_bytes(), "application/x-ndjson",
                                          {"Content-Encoding": "gzip"}, cors=False, compress=False)
                    poses = folder / "poses.f32"
                    if not poses.is_file():
                        return self._error("this replay has no poses (a drawn-body run)", 404, cors=False)
                    return self._send(200, poses.read_bytes(), "application/octet-stream", cors=False, compress=False)
                # every endpoint that reads one fly takes ?fly=k (default 0, the protagonist; docs/TWO_FLIES_PLAN.md 5.7)
                raw = get("fly", "0")
                # a few ASCII digits only: str.isdigit also takes '\u00b2' and the like, which int() refuses, and a
                # thousands-digit string trips int()'s conversion limit (both would answer 500 instead of 404)
                if not (isinstance(raw, str) and raw.isascii() and raw.isdigit() and len(raw) <= 3
                        and int(raw) < len(game.flies)):
                    return self._error(f"no fly {raw[:20]}", 404)
                fly = game.fly(int(raw))
                conn = fly.conn                           # a grown fly swaps the wiring in
                if path == "/api/layout":
                    return self._send(200, game.layout_json if fly.id == 0 else fly.layout_json(), "application/json")
                if path == "/api/types":
                    text = get("q", "")
                    hits = conn.find_types(text, limit=int(get("limit", 50))) if text else []
                    known = set(conn.tables["types"])                   # an alias is marked with what it stands for
                    return self._json({"ok": True, "types": [
                        {"type": t, "n": n, **({"alias_of": conn.aliases[t]} if t in conn.aliases and t not in known else {})}
                        for t, n in hits]})
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
                    info["rate_hz"], parts_role = fly.io.neuron(i)
                    info["inputs"] = conn.inputs_of(f"index:{i}", top=8)
                    info["outputs"] = conn.outputs_of(f"index:{i}", top=8)
                    info["genes"] = genetics.genes_of(conn, i)
                    info["vfb"] = vfb.describe_type(conn.types[i], conn)
                    info["receptors"] = vfb.receptors_of_type(conn.types[i], conn)
                    info["parts"] = parts_role
                    return self._json({"ok": True, "neuron": info})
                if path == "/api/ontology":
                    ont = vfb.ontology_for(conn)
                    if get("id"):
                        info = ont.class_info(get("id"), conn)
                        if info is None:
                            return self._error("no such class among the kit's cell types", 404)
                        return self._json({"ok": True, "class": info})
                    text = get("q", "").strip()
                    try:
                        limit = max(1, min(200, int(get("limit", 30))))
                    except ValueError:
                        return self._error("limit must be a whole number")
                    return self._json({"ok": True, "q": text, "classes": ont.search(text, conn, limit=limit) if text else []})
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
                            members = [int(m) for m in members if fly.has_soma[m]][:1]
                            pts.append(members[0] if members else None)
                        soma_paths.append(pts)
                    return self._json({"ok": True, "from": src, "to": dst, "secs": round(time.time() - t0, 2),
                                       "paths": [p.to_dict() for p in paths], "neurons": soma_paths,
                                       "relays": relay_ranking(paths)[:10]})
                if path == "/api/history":
                    keys = [k for k in get("keys", "").split(",") if k]
                    n = int(get("n", 400))
                    bin_ms, out = fly.history(keys, n)
                    return self._json({"ok": True, "bin_ms": bin_ms, "history": out})
                if path == "/api/learning":
                    info = fly.io.learning_summary() if fly.io.has_plasticity else None
                    if info is None:
                        return self._json({"ok": True, "learning": None})
                    return self._json({"ok": True, "learning": info["mbon"], "settings": info["settings"],
                                       "depressed_fraction": info["depressed_fraction"]})
                if path == "/api/genome":
                    return self._json({"ok": True, "levels": [{"level": lv, "label": lb} for lv, lb in wiring.LEVELS], **fly.genome_status()})
                if path == "/api/genes":
                    return self._json({"ok": True, **fly.genetics})
                if path == "/api/parts":
                    return self._json({"ok": True, "on": fly.parts_on, "tables": fly.parts_list().describe(),
                                       "counts": fly.parts_counts(), "status": fly.io.parts_status()})
                if path == "/api/lines":
                    spec = get("spec", "").strip()
                    if not spec:
                        return self._error("spec is required")
                    try:
                        n = max(1, min(8, int(get("n", "4"))))
                        return self._json({"ok": True, **fly.neuronbridge.lines_for(conn, spec, max_neurons=n)})
                    except ValueError as e:
                        return self._error(str(e))
                    except genetics.NeuronBridgeError as e:
                        return self._error(str(e), 502)
                if path == "/api/driver":
                    line = get("line", "").strip()
                    if not line:
                        return self._error("line is required")
                    try:
                        return self._json({"ok": True, **fly.neuronbridge.neurons_for_line(conn, line)})
                    except ValueError as e:
                        return self._error(str(e))
                    except genetics.NeuronBridgeError as e:
                        return self._error(str(e), 502)
                if path == "/api/decoder":
                    return self._json({"ok": True, "targets": fly.decoder.dn_targets})
                if path == "/api/recording":
                    rec = game.recording or []
                    body = json.dumps({"frames": rec, "settings": fly.io.settings()}).encode()
                    return self._send(200, body, "application/json",
                                      {"Content-Disposition": "attachment; filename=fly-session.json"})
                if path == "/api/spikes":
                    t_ms, idx = fly.io.record("arrays")     # (read under the brain's lock: the game thread appends to it)
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
            try:                                     # drain the body first so a keep-alive connection stays in sync
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length) if length > 0 else b""
            except (ValueError, OSError):
                self.close_connection = True
                return self._send(400, b"bad request", "text/plain")
            if self.path != "/api/action":
                return self._send(404, b"not found", "text/plain")
            try:
                data = json.loads(raw or b"{}")
                if not isinstance(data, dict):
                    raise ValueError("the action must be a JSON object")
                if data.get("type") == "capture":    # writes to disk: only from this server's own page (8.8)
                    why = self._own_page()
                    if why:
                        return self._json({"ok": False, "error": why}, 403, cors=False)
                reply = game.action(data)
            except Exception as e:
                reply = {"ok": False, "error": str(e)}
            self._json(reply)

    return Handler


def serve(game, port: int = 8765, open_browser: bool = True, host: str = "127.0.0.1"):
    server, err = None, None
    last = min(port + 19, 65535)
    for p in range(port, last + 1):
        try:
            server = ThreadingHTTPServer((host, p), make_handler(game))
            break
        except OSError as e:
            if isinstance(e, socket.gaierror) or e.errno == errno.EADDRNOTAVAIL:   # the address, not the port: no port helps
                raise SystemExit(f"Could not listen on {host}: it is not one of this computer's addresses ({e}). "
                                 "Leave out --host to use this computer only: run the same command without it.")
            err = e
    if server is None:
        other = 9000 if not port <= 9000 <= last else 8000             # a range that was not just tried
        raise SystemExit(f"Could not find a free port on {host} from {port} to {last}" + (f" ({err})" if err else "")
                         + f". Try another one: run the same command with --port {other}")   # keeps --female, --body ...
    server.daemon_threads = True
    everywhere = host in ("0.0.0.0", "", "::")
    url = f"http://{'127.0.0.1' if everywhere else host}:{server.server_address[1]}/"
    print(f"\nThe fly is alive at {url}\n(keep this window open; press Ctrl+C here to quit)\n")
    if everywhere:
        print(f"Listening on every network interface: other computers on your network can open "
              f"http://<this computer's address>:{server.server_address[1]}/ and drive the fly (there is no password).\n")
    if open_browser:
        webbrowser.open(url)
    loop = threading.Thread(target=game.loop, daemon=True)
    loop.start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Bye!")
    finally:
        # let the game loop finish its tick and stop before Python shuts down: a thread still inside MuJoCo (the physics
        # body) while the interpreter tears down crashes the process; then let the brain processes go (Game.close), only
        # once no tick can still be using them
        game.stop_loop.set()
        loop.join(timeout=10)
        if loop.is_alive():
            print("The game's last tick is taking more than 10 s; closing the brains anyway.")
        close = getattr(game, "close", None)
        if close is not None:
            close()
        server.server_close()
    return server
