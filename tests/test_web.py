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
    assert {"realBtn", "realDlg", "genomeRealBtn"} <= looked_up
