"""Static checks on the browser page: element ids are unique and every id the scripts look up exists.

A repeated id is silent in the browser: ``getElementById`` returns the first element, so a
second button with the same id never gets its handler and steals the first one's (this is
how the header's "What's real here?" once stopped opening its dialog).
"""
import collections
import pathlib
import re

WEB = pathlib.Path(__file__).resolve().parent.parent / "virtual_fly" / "web"
ID = re.compile(r'\bid="([^"$]+)"')


def _ids():
    """Every literal id in the page and in the scripts' HTML strings, with repeats."""
    ids = ID.findall((WEB / "index.html").read_text(encoding="utf-8"))
    for js in sorted(WEB.glob("*.js")):
        ids += ID.findall(js.read_text(encoding="utf-8"))
    return ids


def test_element_ids_are_unique():
    repeated = sorted(k for k, n in collections.Counter(_ids()).items() if n > 1)
    assert repeated == []


def test_every_looked_up_id_exists():
    known = set(_ids())
    looked_up = set()
    for js in sorted(WEB.glob("*.js")):
        looked_up |= set(re.findall(r'\$\("([^"]+)"\)', js.read_text(encoding="utf-8")))
    assert sorted(looked_up - known) == []
    assert {"realBtn", "realDlg", "genomeRealBtn", "genomeSyn"} <= looked_up


def _rule(css, selector):
    """The declarations of every rule for ``selector`` that starts a line, joined (whitespace-normalised)."""
    found = re.findall(r"(?m)^\s*" + re.escape(selector) + r"\s*\{([^}]*)\}", css)
    return " ".join(" ".join(found).split()) or None


def test_the_neuron_popover_is_not_clipped_by_its_card():
    # every card is paint-contained (no reflow shakes the sidebar), which clips what overflows it; the brain
    # card's popover is taller than the card, and its links and buttons were cut off with it. Layout containment
    # kept it out of the sidebar's scroll range, out of reach when the brain card was the last or the only card
    css = (WEB / "style.css").read_text(encoding="utf-8")
    assert "paint" in _rule(css, ".card")
    brain = _rule(css, "#brainCard")
    assert brain and "contain: none" in brain and "paint" not in brain and "z-index" in brain


def test_the_header_fits_narrow_screens():
    # the header's buttons wrap to a second line instead of running off a phone's or a tablet's screen,
    # and the round '?' button keeps its size instead of being squashed
    css = (WEB / "style.css").read_text(encoding="utf-8")
    assert "flex: none" in _rule(css, ".linkbtn.round")
    narrow = css[css.index("@media (max-width: 1080px)"):]
    assert "flex-wrap: wrap" in _rule(narrow, "header")


def test_the_panels_menu_keeps_the_focus_between_ticks():
    # a checkbox in the Panels menu that dropped the focus closed the menu (it closes when the focus leaves)
    js = (WEB / "layout.js").read_text(encoding="utf-8")
    menu = js[js.index("  buildMenu() {"):js.index("  refreshMenu() {")]
    assert ".blur()" not in menu
    assert "menu.tabIndex = -1" in js


def test_the_why_line_wraps_to_two_lines_and_keeps_its_full_text_as_a_tooltip():
    # the water and feeding-bout explanations are longer than the card is wide: two lines show (the second one
    # also under the mode chip), the tooltip has it all,
    # and the card keeps one fixed height (it changes forty times a second above every other panel)
    css = (WEB / "style.css").read_text(encoding="utf-8")
    why, mode = _rule(css, ".why"), _rule(css, ".why .mode")
    assert "height: 40px" in why and "line-height: 20px" in why and "-webkit-line-clamp: 2" in why and "nowrap" not in why
    assert "display: inline-block" in mode and "white-space: nowrap" in mode and "line-height: 18px" in mode
    js = (WEB / "panels.js").read_text(encoding="utf-8")
    update = js[js.index("class WhyPanel"):js.index("class KeyNeurons")]
    assert "drv.title = why" in update


def test_the_two_fly_controls_exist_and_no_verdict_is_written():
    # two flies (docs/TWO_FLIES_PLAN.md 5.8): a fly menu in the header, a second brain canvas kept alive beside the first
    # (a WebGL context is never created and destroyed on a focus switch), and her decision neurons are shown as readouts,
    # never called acceptance or rejection anywhere on the page
    ids = set(_ids())
    assert {"focusSel", "brain", "brain2", "brainOverlay", "brainOverlay2", "realOther"} <= ids
    text = (WEB / "index.html").read_text(encoding="utf-8") + "".join(js.read_text(encoding="utf-8") for js in sorted(WEB.glob("*.js")))
    assert not re.search(r"accept(ance|ed)|reject(ion|ed)", text, re.I)
    css = (WEB / "style.css").read_text(encoding="utf-8")
    assert _rule(css, "#brain2") and "display: block" in _rule(css, "#brain2") and "display: none" in _rule(css, "#brain[hidden], #brain2[hidden], #brainOverlay[hidden], #brainOverlay2[hidden]")
    js = (WEB / "app.js").read_text(encoding="utf-8")
    assert "setActionFly" in js and "posesOf" in js and "flyPose(" not in js and "femalePose(" not in js


def _js(name):
    return (WEB / name).read_text(encoding="utf-8")


def test_the_page_asks_the_focused_fly_for_its_types_ontology_and_routes():
    # two flies: the lab's live type search, the ontology search and the pathway explorer name the fly in focus
    # (api/types, api/ontology and api/trace take ?fly=k), so the female's wiring is searched when she is in focus,
    # and a route traced in one fly's wiring is cleared when the other fly's map takes its place
    js = _js("panels.js")
    assert re.search(r"api/types\?q=[^`]*\$\{this\.flyQ\(\)\}", js)
    assert re.search(r"api/ontology\?q=[^`]*\$\{this\.flyQ\(\)\}", js)
    assert re.search(r"api/trace\?from=[^`]*\$\{fly\}", js) and "fly_id ? `&fly=${this.brain.L.fly_id}`" in js
    assert "refocus(brain) {" in js and "this.brain.setPath(null)" in js
    assert "panels.paths.refocus(brain)" in _js("app.js")
    assert "if (onto && !this.ontoWired) this.wireOntology();" in js


def test_the_walking_urge_control_follows_the_focused_fly():
    # each fly's state entry carries its own `autopilot` (docs/API.md "Two flies"): the checkbox shows the focused fly's,
    # and the W key toggles it from its own value, not from fly 0's; the fly menu lets go of the keyboard after a choice
    js = _js("app.js")
    assert "if (document.activeElement !== ap) ap.checked = !!s.autopilot;" in js and "!focus) ap.checked" not in js
    assert "on: !(V && V.autopilot)" in js and "on: !(S && S.autopilot)" not in js
    assert re.search(r'\$\("focusSel"\)\.onchange = \(e\) => \{[^\n]*e\.target\.blur\(\)', js)


def test_a_page_local_message_outlives_the_next_tick():
    # the server's `msg` arrives with every tick and is empty most of the time; a message the page itself shows
    # (the F key with a simulated partner) is held for a few seconds instead of being wiped 25 ms later
    js = _js("app.js")
    assert "toastHold" in js and "function toast(msg, holdMs = 0)" in js
    assert re.search(r'toast\("The female is simulated here[^"]*", 3000\)', js)


def test_the_male_page_keeps_its_neuronbridge_tooltips():
    # the two NeuronBridge buttons are disabled for a female (MaleCNS bodies only); on a male page they keep the
    # tooltips index.html gives them instead of an empty one
    html = (WEB / "index.html").read_text(encoding="utf-8")
    assert 'id="lineBtn" title="Which MaleCNS neurons does this line label? (NeuronBridge, needs internet)"' in html
    assert 'id="linesBtn" title="Which driver lines label these neurons? (NeuronBridge, needs internet)"' in html
    js = _js("panels.js")
    assert "b.dataset.title = b.title" in js and ": b.dataset.title;" in js
    assert not re.search(r'\.title = this\.sex === "female" \? "[^"]*" : "";', js)


def test_a_hidden_brain_map_keeps_a_bounded_spike_buffer():
    # in the 2-D fallback the map not in focus (the other fly's) is never drawn, so its pending spike lists are capped
    js = _js("brain3d.js")
    assert re.search(r"const PENDING_MAX = \d+;", js)
    assert "if (this.pending.length >= PENDING_MAX) this.pending.shift();" in js


def test_her_cues_are_labelled_hand_built_on_screen_and_events_say_whose_they_are():
    # plan 1.5 rule 2: the marks at her abdomen are hand-built cues with provisional thresholds, said under the dish
    # and on the 'Her decisions' group; with two flies each per-fly event row starts with whose it is
    html = (WEB / "index.html").read_text(encoding="utf-8")
    assert 'id="cuesNote" hidden' in html and "hand-built cues, provisional" in html
    js = _js("panels.js")
    assert "const CUES_NOTE" in js and "hand-built cues" in js and 'if (g === "Her decisions") grp.title = CUES_NOTE;' in js
    assert 'class="who"' in js and "S.flies.find((x) => x.id === e.fly)" in js
    assert 'setShown($("cuesNote"), s.flies.some((f) => f.sex === "female"))' in _js("app.js")
    css = (WEB / "style.css").read_text(encoding="utf-8")
    assert _rule(css, ".events li .who") and _rule(css, ".rows .grp .note") and _rule(css, ".cuesnote")


def test_the_3d_dish_is_there_and_fetches_three_only_when_asked():
    # docs/TWO_FLIES_PLAN.md 7.1, 7.3, 7.4: a 3-D canvas over the stage, a toggle beside the zoom button, the badge that says
    # the legs are an animation and whose body model the flies wear, the camera presets, a message box for the fallback;
    # three.js (the import map's "three") and arena3d.js are imported lazily, never statically, so the default page fetches
    # nothing under vendor/three/ or models/
    html = (WEB / "index.html").read_text(encoding="utf-8")
    ids = set(_ids())
    assert {"arena3d", "view3dBtn", "badge3d", "cam3d", "msg3d"} <= ids
    assert '<canvas id="arena3d" hidden></canvas>' in html and 'id="badge3d" hidden' in html
    assert "3-D animation: the legs follow the gait phase; not physics. Both flies use NeuroMechFly's body, built from a female fly; the legs replay NeuroMechFly's recorded stride" in html
    assert '"three": "./vendor/three/three.module.js"' in html and '"three/addons/": "./vendor/three/addons/"' in html
    assert html.index('type="importmap"') < html.index('type="module" src="app.js"')
    for js in sorted(WEB.glob("*.js")):
        text = js.read_text(encoding="utf-8")
        assert not re.search(r'(?m)^\s*import\b[^\n]*\bfrom\s*["\']three', text), js.name      # no static import of three anywhere
        if js.name != "arena3d.js":
            assert 'import("three' not in text, js.name
    a3 = _js("arena3d.js")
    assert 'await import("three")' in a3 and 'await import("three/addons/loaders/GLTFLoader.js")' in a3 and 'await import("three/addons/controls/OrbitControls.js")' in a3
    assert '"models/nmf_fly.glb"' in a3 and '"models/nmf_gait.json"' in a3 and '"models/nmf_gait.bin"' in a3
    assert "webgl2" in a3 and "webglcontextlost" in a3 and "webglcontextrestored" in a3
    assert "hand-built" in a3.lower() and "not physics" in a3
    app = _js("app.js")
    assert 'await import("./arena3d.js")' in app and 'from "./arena3d.js"' not in app
    assert "function activeArena()" in app and 'guard("the 3-D dish"' in app and 'onDish("pointerdown"' in app
    css = (WEB / "style.css").read_text(encoding="utf-8")
    assert "position: absolute" in _rule(css, "#arena3d") and "inset: 0" in _rule(css, "#arena3d")
    assert "z-index: 2" in _rule(css, ".retina") and "z-index: 2" in _rule(css, ".badge3d") and "z-index: 3" in _rule(css, ".toast")
    # the vendored files the import map points at exist, with their licence and version record
    for name in ("three.module.js", "three.core.js", "LICENSE", "VERSION.txt", "addons/loaders/GLTFLoader.js", "addons/controls/OrbitControls.js"):
        assert (WEB / "vendor" / "three" / name).is_file(), name


def test_the_badge_shows_the_frame_rate():
    """The 3-D badge carries a frame-rate readout the dish updates once a second (plan 7.5: measured, never promised)."""
    html = (WEB / "index.html").read_text()
    assert 'id="fps3d"' in html and "fps" in html
    js = (WEB / "arena3d.js").read_text()
    assert "this.ui.fps" in js and "fps3d" in (WEB / "app.js").read_text()


def test_the_3d_view_says_what_is_hand_built_in_whats_real(conn):
    # plan 1.5 rule 2: the animation, the female body model for both flies, the recorded stride and the drawn scale are
    # in the "What's real here?" hand-built list of every fly, and the words acceptance/rejection never appear in it
    from virtual_fly.game import Game
    from virtual_fly.settings import build_brain
    g = Game(build_brain(conn, "game", seed=0), autopilot=False, seed=1)
    try:
        hb = g.whats_real()["hand_built"]
        text = " ".join(hb)
        assert "an animation, not physics" in text and "NeuroMechFly's recorded stride" in text
        assert "built from a female fly" in text and "about three times real size" in text
        assert sum("When the 3-D view is on" in h for h in hb) == 4
        assert not re.search(r"accept(ance|ed)|reject(ion|ed)", text, re.I)
    finally:
        g.close()


def test_the_replay_player_is_there():
    """Record-and-replay in the page (docs/TWO_FLIES_PLAN.md 8.8): the Recording card's save switch and replay list, the player
    and its badge in the stage, the recorded poses driving the 3-D view, the live ticks kept aside while a replay plays."""
    html = (WEB / "index.html").read_text(encoding="utf-8")
    for i in ("capBtn", "capInfo", "replayList", "replayRefresh", "player", "plPlay", "plSpeed", "plScrub", "plTime", "plLeave",
              "replayBadge", "badge3dText"):
        assert f'id="{i}"' in html, i
    assert "replay of a recorded run" in html
    app = (WEB / "app.js").read_text(encoding="utf-8")
    assert "api/replay/" in app and "function replayLoad" in app and 'get("replay")' in app
    assert "if (replay.active) return;" in app                       # the live ticks are kept, not shown, while a replay plays
    assert "replayPoses" in (WEB / "arena3d.js").read_text(encoding="utf-8")
    panels = (WEB / "panels.js").read_text(encoding="utf-8")
    assert 'type: "capture"' in panels and "api/replays" in panels
