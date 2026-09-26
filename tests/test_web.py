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
    # card's popover is taller than the card, and its links and buttons were cut off with it
    css = (WEB / "style.css").read_text(encoding="utf-8")
    assert "paint" in _rule(css, ".card")
    brain = _rule(css, "#brainCard")
    assert brain and "contain: layout" in brain and "paint" not in brain and "z-index" in brain


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
