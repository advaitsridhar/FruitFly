"""Where the kit keeps its data (a source checkout against an installed copy), and the connectome download's
messages: a damaged file already in place, and Ctrl+C part-way (no network: urlopen is replaced)."""

import io
import os
import shutil
import subprocess
import sys
import urllib.error

import pytest

import virtual_fly.connectome as C


def test_a_checkout_keeps_its_data_in_data():
    from virtual_fly import parts, vfb
    assert not C.INSTALLED
    assert C.SHIPPED_DATA_DIR == C.PROJECT_DIR / "data" == vfb.DATA_DIR == parts.REGION_FILE.parent
    if not os.environ.get("FLY_DATA_DIR"):
        assert C.DEFAULT_DATA_FILE == C.PROJECT_DIR / "data" / "malecns-v1.0.flyb.gz"
    pyproject = (C.PROJECT_DIR / "pyproject.toml").read_text(encoding="utf-8")
    for f in (vfb.MAP_FILE, vfb.TREE_FILE, vfb.RX_FILE, parts.REGION_FILE):
        assert f.exists() and f'"{f.name}"' in pyproject                    # and the wheel ships each of them


def test_an_installed_copy_reads_its_own_files_and_downloads_to_a_cache(tmp_path):
    from virtual_fly import parts, vfb
    site = tmp_path / "site-packages"                                       # what `pip install .` lays out
    shutil.copytree(C.PACKAGE_DIR, site / "virtual_fly", ignore=shutil.ignore_patterns("__pycache__", "web"))
    (site / "virtual_fly" / "data").mkdir()
    for f in (vfb.MAP_FILE, vfb.TREE_FILE, vfb.RX_FILE, parts.REGION_FILE):
        shutil.copy(f, site / "virtual_fly" / "data")
    code = ("import virtual_fly.connectome as c, virtual_fly.vfb as v, virtual_fly.parts as p; "
            "print(c.INSTALLED, c.DEFAULT_DATA_FILE.parent, v.MAP_FILE.parent == c.PACKAGE_DIR / 'data', "
            "not v.ontology().empty, p.region_table() is not None)")
    env = {k: v for k, v in os.environ.items() if k not in ("FLY_DATA_DIR", "FLY_DATA_FILE")}
    env.update(PYTHONPATH=str(site), HOME=str(tmp_path / "home"), USERPROFILE=str(tmp_path / "home"))

    def run():
        return subprocess.run([sys.executable, "-c", code], env=env, cwd=tmp_path, capture_output=True, text=True,
                              check=True).stdout.split()
    assert run() == ["True", str(tmp_path / "home" / ".cache" / "virtual-fly"), "True", "True", "True"]
    env["FLY_DATA_DIR"] = str(tmp_path / "shared")
    assert run()[1] == str(tmp_path / "shared")


def test_a_damaged_file_is_named_when_the_download_fails(tmp_path, monkeypatch, capsys):
    target = tmp_path / "malecns-v1.0.flyb.gz"
    target.write_bytes(b"\x1f\x8b" + bytes(998))                             # say, a download cut short

    def blocked(url, timeout):
        raise urllib.error.URLError("blocked by a firewall")
    monkeypatch.setattr(C.urllib.request, "urlopen", blocked)
    with pytest.raises(SystemExit) as e:
        C.download_connectome(target)
    assert "is damaged or is not the right file (1,000 bytes; the right one has 22,964,094" in capsys.readouterr().err
    assert "blocked by a firewall" in str(e.value) and "in place of the damaged file there" in str(e.value)
    assert target.stat().st_size == 1000 and not target.with_suffix(".part").exists()


def test_ctrl_c_during_the_download_leaves_no_part_file(tmp_path, monkeypatch):
    target = tmp_path / "malecns-v1.0.flyb.gz"

    class Response(io.BytesIO):
        headers = {"Content-Length": str(C.DATA_BYTES)}
        reads = 0

        def read(self, n=-1):
            self.reads += 1
            if self.reads > 1:
                raise KeyboardInterrupt                                      # Ctrl+C after the first megabyte
            return bytes(1 << 20)
    monkeypatch.setattr(C.urllib.request, "urlopen", lambda url, timeout: Response())
    with pytest.raises(SystemExit, match="Download stopped"):
        C.download_connectome(target, quiet=True)
    assert not target.exists() and not target.with_suffix(".part").exists()


def test_ctrl_c_while_the_female_fly_is_built(monkeypatch):
    from virtual_fly import flywire

    def interrupted(quiet=False):
        raise KeyboardInterrupt
    monkeypatch.setattr(flywire, "ensure_female", interrupted)
    with pytest.raises(SystemExit, match="Stopped before the female fly was built"):
        C.load_connectome(female=True, quiet=True)
