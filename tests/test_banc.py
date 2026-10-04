"""BANC, the female fly with a nerve cord (virtual_fly/banc.py; docs/TWO_FLIES_PLAN.md section 9): the builder on a tiny
BANC-like table written here with pyarrow, the file it writes read back by Connectome, the loader's dataset switch, the
gain from the meta, and the command lines' refusals. No download: the real deposit is pinned by hash and fetched on use."""
import gzip
import json
import struct

import numpy as np
import pytest

from virtual_fly import banc, connectome as C, flywire
from virtual_fly.brain import DEFAULT_GAIN, FlyBrain
from virtual_fly.connectome import Connectome

pyarrow = pytest.importorskip("pyarrow", reason="BANC's files are Arrow tables (the female extra)")
import pyarrow.feather as feather  # noqa: E402

META_ROWS = [   # root_id, proofread, roughly, status, side, region, nerve, neuromere, super_class, cell_class, sub, type, fafb, manc, malecns, match, dimorphic, nt, score, verified, pos
    ("101", "TRUE", "FALSE", None, "left", "central_brain", None, None, "central_brain_intrinsic", "kenyon_cell", None, "KCg-m", "KCg-m", None, "KCg-m", "4001", "isomorphic", "acetylcholine", "0.9", "acetylcholine", "100, 200, 300"),
    ("102", "FALSE", "TRUE", None, "right", "central_brain", None, None, "descending", "descending_neuron", None, "DNp01", "DNp01", "DNp01", "DNp01", "4002", "dimorphic", "acetylcholine", "0.3", None, "110, 210, 310"),
    ("103", "TRUE", "FALSE", None, "left", "ventral_nerve_cord", "left_prothoracic_leg_nerve", "T1", "motor", "leg_motor_neuron", None, "MNfl01", None, "MNfl01", None, None, "female-specific", "glutamate", "0.8", "glutamate", "120, 900, 320"),
    ("104", "TRUE", "FALSE", None, "right", "central_brain", None, None, "sensory", "bristle_neuron", None, None, None, None, "BM_InOm", "4004", "isomorphic", "tyramine", "0.95", None, "130, 230, 330"),
    ("105", "TRUE", "FALSE", "GLIA,NOT_A_NEURON", "left", "central_brain", None, None, "glia", None, None, None, None, None, None, None, None, None, None, None, "1, 2, 3"),
    ("106", "FALSE", "FALSE", "TOO_SMALL", "left", "optic_lobe", None, None, "optic_lobe_intrinsic", None, None, None, None, None, None, None, None, "gaba", "0.9", None, None),
    ("107", "TRUE", "FALSE", None, "right", "optic_lobe", None, None, "glia", "astrocyte", None, None, None, None, None, None, None, None, None, None, "5, 6, 7"),
    ("101", "FALSE", "TRUE", None, "left", "central_brain", None, None, "central_brain_intrinsic", "kenyon_cell", None, "KCg-m", "KCg-m", None, "KCg-m", "4001", "isomorphic", "dopamine", "0.9", None, "100, 200, 300"),
    ("108", "TRUE", "FALSE", None, "left", "ventral_nerve_cord", "abdominal_nerve_trunk", "A1", "ventral_nerve_cord_intrinsic", "transverse_neuron", "ventral_nerve_cord_bilateral_interconnecting", None, None, "IN01A001", "IN01A001", None, "isomorphic", "gaba", "0.4", None, "140, 950, 340"),
]
COLS = ("root_id", "proofread", "roughly_proofread", "status", "side", "region", "nerve", "neuromere", "super_class", "cell_class",
        "cell_sub_class", "cell_type", "fafb_cell_type", "manc_cell_type", "malecns_cell_type", "malecns_match", "sexually_dimorphic",
        "neurotransmitter_predicted", "neurotransmitter_score", "neurotransmitter_verified", "root_position_nm")
EDGES = [("101", "102", 5), ("102", "103", 3), ("103", "104", 1), ("104", "101", 7), ("105", "101", 9), ("101", "101", 2), ("108", "103", 4), ("106", "108", 1)]


@pytest.fixture
def sources(tmp_path):
    meta = {c: [r[i] for r in META_ROWS] for i, c in enumerate(COLS)}
    feather.write_feather(pyarrow.table(meta), str(tmp_path / "banc_888_meta.feather"))
    edges = {"pre": [e[0] for e in EDGES], "post": [e[1] for e in EDGES], "count": pyarrow.array([e[2] for e in EDGES], pyarrow.int32()),
             "norm": [0.1] * len(EDGES), "post_count": [1] * len(EDGES), "pre_count": [1] * len(EDGES)}
    feather.write_feather(pyarrow.table(edges), str(tmp_path / "banc_888_edgelist_simple_v3.feather"))
    return tmp_path


class FakeMale:
    """The male file as vote_maps reads it: body ids and their class, subclass and nerve."""
    body_id = np.array([4001, 4002, 4004])
    cls = np.array(["Kenyon_Cell", "", "mechanosensory_tactile"])
    subclass = np.array(["g", "", "head"])
    nerve = np.array(["", "", "AN"])


def test_the_selection_the_maps_and_the_rows(sources):
    meta = banc.read_meta(sources / "banc_888_meta.feather")
    assert len(meta) == 9 and meta[0]["root_id"] == "101"
    kept, counts = banc.select_neurons(meta)
    assert sorted(int(r["root_id"]) for r in kept) == [101, 102, 103, 104, 108]
    assert counts == {"rows": 9, "glia_trachea_or_not_a_neuron": 2, "not_proofread": 1, "duplicate_root_ids": 1, "kept": 5}
    assert next(r for r in kept if r["root_id"] == "101")["proofread"] == "TRUE"       # the proofread row of the duplicate wins
    maps = banc.vote_maps(kept, FakeMale())
    assert maps["cls"]["kenyon_cell"] == {"to": "Kenyon_Cell", "share": 1.0, "votes": 1}
    assert maps["cls"]["bristle_neuron"]["to"] == "mechanosensory_tactile" and maps["nerve"] == {}
    rows = banc.neuron_rows(kept, maps)
    by = {r["root"]: r for r in rows}
    assert by[101]["cls"] == "Kenyon_Cell" and by[101]["superclass"] == "cb_intrinsic" and by[101]["side"] == "L"
    assert by[102]["nt"] == "unclear" and by[102]["sign_nt"] == "acetylcholine"       # scored 0.3: unclear, the sign kept
    assert by[102]["dimorphism"] == "sexually dimorphic" and by[102]["superclass"] == "descending_neuron"
    assert by[103]["superclass"] == "vnc_motor" and by[103]["neuromere"] == "T1" and by[103]["dimorphism"] == "female-specific"
    assert by[103]["nerve"] == "" and by[103]["cls"] == "leg_motor_neuron"            # no vote: BANC's own words stay
    assert by[104]["nt"] == "tyramine" and by[104]["cls"] == "mechanosensory_tactile" and by[104]["soma"] == [130.0, 230.0, 330.0]
    assert by[104]["type"] == "BM_InOm" and by[104]["superclass"] == "cb_sensory"       # no BANC name: the MaleCNS one it matches
    assert by[108]["superclass"] == "vnc_intrinsic" and by[108]["type"] == "IN01A001"   # the MANC name when BANC has no cell_type
    assert banc.neuron_rows(kept, {})[0]["cls"] == "kenyon_cell"                       # an empty map: the data's own words
    assert banc.known_transmitters(kept) == {"KCg-m": {"nt": ["acetylcholine"], "source": "BANC neurotransmitter_verified (Bates et al. 2026)"},
                                            "MNfl01": {"nt": ["glutamate"], "source": "BANC neurotransmitter_verified (Bates et al. 2026)"}}
    assert banc.data_aliases(kept, ["BM_InOm", "KCg-m", "NOPE"]) == {"BM_InOm": "body:104"}   # KCg-m is a BANC type already; NOPE nowhere


def test_the_file_written_reads_back_with_the_cord(sources, tmp_path):
    meta = banc.read_meta(sources / "banc_888_meta.feather")
    kept, counts = banc.select_neurons(meta)
    pre, post, syn = banc.read_edges(sources / "banc_888_edgelist_simple_v3.feather")
    assert pre.dtype == np.int64 and syn.tolist() == [5, 3, 1, 7, 9, 2, 4, 1]
    ids = np.array(sorted(int(r["root_id"]) for r in kept))
    keep = np.isin(pre, ids) & np.isin(post, ids) & (pre != post)
    assert int((~keep).sum()) == 3                                                   # one to glia, one self-connection, one from a dropped id
    rows = banc.neuron_rows(kept, banc.vote_maps(kept, FakeMale()))
    meta_out = {"dataset": banc.DATASET, "sex": "female", "has_vnc": True, "default_gain": banc.DEFAULT_GAIN,
                "build": banc.BUILD, "aliases": {}, "nt_signs": banc.NT_SIGN, "layout_axis_hint": "y"}
    path = tmp_path / "banc-test.flyb.gz"
    flywire.write_flyb(path, rows, pre[keep], post[keep], syn[keep], meta_out, dataset=banc.DATASET, nt_sign=banc.NT_SIGN)
    c = Connectome(path)
    assert c.dataset == "banc:v888" and c.sex == "female" and c.n == 5 and c.n_edges == 5
    assert c.meta["has_vnc"] is True and c.meta["default_gain"] == 1.0 and c.meta["layout_axis_hint"] == "y"
    assert c.select("superclass:vnc_motor").size == 1 and c.select("superclass:vnc_intrinsic").size == 1
    assert c.select("neuromere:T1").size == 1 and c.select("nt:tyramine").size == 1
    assert int(c.sign[c.select("nt:tyramine")[0]]) == banc.TYRAMINE_SIGN == 1          # decision 31: +1, a switch
    assert int(c.sign[c.select("DNp01")[0]]) == 1 and c.nt[c.select("DNp01")[0]] == "unclear"
    assert c.select("gene:fru").size == 0 and c.select("dimorphism:female").size == 1
    assert banc.built_with(path) == banc.BUILD
    assert "tyramine" in c.tables["nts"]


def test_the_loader_takes_a_dataset(tmp_path, monkeypatch, conn):
    monkeypatch.setattr(banc, "ensure_banc", lambda quiet=False: tmp_path / "missing.flyb.gz")
    with pytest.raises(ValueError, match="dataset must be one of"):
        C.load_connectome(dataset="nope")
    with pytest.raises(ValueError, match="female=True means"):
        C.load_connectome(female=True, dataset="banc")
    with pytest.raises(ValueError, match="a path or a dataset"):
        C.load_connectome(path=tmp_path / "x", dataset="banc")
    with pytest.raises(SystemExit, match="no connectome file"):
        C.load_connectome(dataset="banc", quiet=True)
    assert "banc" in C.DATASETS and "flywire" in C.DATASETS
    damaged = tmp_path / "banc-v888.flyb.gz"
    damaged.write_bytes(b"not gzipped")
    monkeypatch.setattr(banc, "BANC_FILE", damaged)
    with pytest.raises(SystemExit, match="BANC fly's file .*damaged.*banc-src"):
        C.load_connectome(dataset="banc", quiet=True)


def test_the_gain_comes_from_the_meta_when_the_file_names_one(conn):
    assert FlyBrain(conn, dt=0.5).gain == DEFAULT_GAIN["male"]                        # the male file names none
    assert "default_gain" not in conn.meta
    class WithGain:                                                                  # a BANC-like file: its own default
        pass
    conn.meta["default_gain"] = 1.7
    try:
        assert FlyBrain(conn, dt=0.5).gain == 1.7
        assert FlyBrain(conn, dt=0.5, gain=0.3).gain == 0.3                            # --gain still wins
    finally:
        del conn.meta["default_gain"]
    assert FlyBrain(conn, dt=0.5).gain == DEFAULT_GAIN["male"]


def test_the_command_lines_refuse_a_fly_they_do_not_know(monkeypatch):
    from virtual_fly import cli, play
    monkeypatch.setattr(cli, "load_connectome", lambda **k: pytest.fail("loaded the data first"))
    monkeypatch.setattr(play, "load_connectome", lambda **k: pytest.fail("loaded the data first"))
    for module in (cli, play):
        for argv in (["--fly", "nope"], ["--female", "--fly", "banc"]):
            with pytest.raises(SystemExit) as e:
                module.main(argv)
            assert e.value.code == 2
    with pytest.raises(SystemExit) as e:
        play.main(["--partner", "cat"])
    assert e.value.code == 2
    assert cli.fly_flag("banc") == " --fly banc" and cli.fly_flag("flywire") == " --female" and cli.fly_flag("male") == ""


def test_genetics_speaks_of_the_dataset(conn):
    from virtual_fly import genetics
    class Banc:
        dataset, sex, meta = "banc:v888", "female", {}
    class Fw:
        dataset, sex = "flywire:v783", "female"
    assert genetics.dataset_kind(conn) == "male" and genetics.dataset_kind(Banc()) == "banc" and genetics.dataset_kind(Fw()) == "flywire"
    assert "no fruitless or doublesex columns" in genetics.SOURCE_BANC
    with pytest.raises(ValueError, match="BANC fly"):
        genetics._malecns_only(Banc())
