"""The female fly (flywire.py): FlyWire's vocabulary, the FLYB writer, the name aliases, the paper's gain, and
experiments on a fly that lacks some of their neurons. A tiny hand-made female brain stands in for FlyWire."""
import numpy as np
import pytest

from virtual_fly import flywire, genetics
from virtual_fly.brain import DEFAULT_GAIN, FlyBrain
from virtual_fly.connectome import Connectome
from virtual_fly.experiments import Experiment, R, run_experiment

SUGAR = flywire.SHIU_SUGAR[:3]                 # three of the published model's sugar cells


def _ann(root, ctype, sc="central", side="left", nt="acetylcholine", cls="", sub="", dim="isomorphic", fd="",
         soma=None, conf="0.9", known=""):
    soma = soma or (100 + 37 * (root % 97), 200 + 23 * (root % 89), 30 + root % 83)     # spread out, for the map
    return {"root_id": str(root), "cell_type": ctype, "hemibrain_type": "", "super_class": sc, "cell_class": cls,
            "cell_sub_class": sub, "side": side, "top_nt": nt, "top_nt_conf": conf, "known_nt": known,
            "known_nt_source": "Davis et al., 2020 (TAPIN)" if known else "", "dimorphism": dim, "fru_dsx": fd,
            "nerve": "", "soma_x": str(soma[0]), "soma_y": str(soma[1]), "soma_z": str(soma[2])}


def _female(tmp_path, min_syn=1, more=()):
    ann = [_ann(SUGAR[0], "LB3", "sensory", cls="gustatory", sub="sugar/water"),
           _ann(SUGAR[1], "LB3", "sensory", "right", cls="gustatory", sub="sugar/water"),
           _ann(SUGAR[2], "LB3", "sensory", cls="gustatory", sub="sugar/water"),
           _ann(11, "LB3", "sensory", cls="gustatory", sub="sugar/water"),       # a water cell: not in the sugar list
           _ann(20, "LB1a,LB1d", "sensory", cls="gustatory", sub="bitter"),
           _ann(21, "LB1b", "sensory", "right", cls="gustatory", sub="bitter"),
           _ann(30, "CB0616"), _ann(31, "CB0616", side="right"),                 # G2N-1
           _ann(40, "CB0219", nt="glutamate"), _ann(41, "CB0219", side="right", nt="glutamate"),
           _ann(50, "CB0701", "motor"), _ann(51, "CB0701", "motor", "right"),    # MN9
           _ann(60, "pC1a", fd="dsx", dim="sexually dimorphic"), _ann(61, "pC1b", side="right", fd="dsx"),
           _ann(70, "vpoDN", "descending", fd="fru", dim="female-specific"),
           _ann(71, "aIP-g", fd="coexpress", dim="potentially female-specific"),
           _ann(80, "R7", "sensory", cls="visual", soma=("", "", "")),
           _ann(90, "lLN1", cls="ALLN", nt="gaba"), _ann(91, "PAM01", cls="DAN", nt="dopamine"), *more]
    pairs = [(SUGAR[0], 30, 80), (SUGAR[1], 31, 80), (SUGAR[2], 30, 80), (30, 50, 80), (31, 51, 80),
             (20, 40, 80), (21, 41, 80), (40, 50, 60), (41, 51, 60), (60, 70, 30), (61, 71, 3),
             (70, 99, 12)]                      # 99: connected, but not in the annotations
    pre, post, syn = (np.array(x, dtype=np.int64) for x in zip(*pairs))
    keep = syn >= min_syn
    pre, post, syn = pre[keep], post[keep], syn[keep]
    extra = sorted({int(x) for x in np.concatenate([pre, post])} - {int(a["root_id"]) for a in ann})
    rows = flywire.neuron_rows(ann, extra)
    meta = {"sex": "female", "aliases": flywire.aliases({r["root"] for r in rows}),
            "known_nt": flywire.known_transmitters(ann)}
    return Connectome(flywire.write_flyb(tmp_path / "female.flyb.gz", rows, pre, post, syn, meta))


@pytest.fixture
def female(tmp_path):
    return _female(tmp_path)


def test_rows_use_the_kits_vocabulary():
    rows = flywire.neuron_rows([_ann(1, "LC4", "visual_projection", side="right", soma=(100, 200, 30)),
                                _ann(2, "R7", "sensory", cls="visual", soma=("", "", "")),
                                _ann(3, "", "descending", dim="female-specific", fd="fru")], extra_ids=[9])
    r = {x["root"]: x for x in rows}
    assert (r[1]["superclass"], r[1]["side"]) == ("visual_projection", "R")
    assert r[2]["superclass"] == "ol_sensory" and np.isnan(r[2]["soma"]).all()
    assert r[1]["soma"] == [400.0, 800.0, 1200.0]                     # FlyWire voxels are 4 x 4 x 40 nm
    assert r[1]["dimorphism"] == "" and r[3]["dimorphism"] == "female-specific" and r[3]["frudsx"] == "fru"
    assert r[3]["superclass"] == "descending_neuron" and r[9]["type"] == "" and r[9]["nt"] == "unclear"


def test_the_file_round_trips(female):
    c = female
    assert c.sex == "female" and c.dataset == flywire.DATASET
    assert c.n == 20 and c.n_edges == 12 and int(c.n_syn.sum()) == 725
    i = c.select("CB0219")
    assert (c.sign[i] == -1).all() and set(c.nt[i]) == {"glutamate"}
    assert c.types[c.select("body:99")][0] == ""                      # connected but unannotated: kept
    assert (c.hex1 == -1).all()                                        # no medulla columns in the annotations


def test_min_synapses_cuts_weak_connections(tmp_path):
    c = _female(tmp_path, min_syn=5)
    assert c.n_edges == 11 and c.synapses_between("pC1b", "aIP-g") == 0


def test_the_kits_names_find_flywires_cells(female):
    c = female
    assert set(c.types[c.select("MN9")]) == {"CB0701"} and c.select("MN9").size == 2
    assert c.select("MN9/L").size == 1 and c.side[c.select("MN9/L")][0] == "L"
    assert set(c.types[c.select("GNG232")]) == {"CB0616"} and set(c.types[c.select("GNG087")]) == {"CB0219"}
    sugar = c.select("LB3b,LB3c")
    assert sorted(c.body_id[sugar].tolist()) == sorted(SUGAR)          # the published model's cells, not all of LB3
    assert c.select("LB3").size == 4
    assert c.select("LB1a,LB1b,LB1c,LB1d").size == 2
    assert set(c.types[c.select("prefix:pC1_")]) == {"pC1a", "pC1b"}
    assert c.select("pIP10").size == 0 and c.select("LgLG3").size == 0 and c.select("MN9,!MN9/R").size == 1


def test_a_real_type_wins_over_an_alias(female):
    c = female
    c.aliases["CB0616"] = "CB0701"                                     # nonsense alias for a name that is a type
    c._cache.clear()
    assert set(c.types[c.select("CB0616")]) == {"CB0616"}
    c.aliases["loopA"], c.aliases["loopB"] = "loopB", "loopA"         # a cycle resolves to nothing, not forever
    assert c.select("loopA").size == 0


def test_genetics_reads_flywires_labels(female):
    c = female
    assert c.select("gene:fru").size == 2 and c.select("gene:dsx").size == 3 and c.select("gene:both").size == 1
    assert c.select("dimorphism:female").size == 2 and c.select("dimorphism:any").size == 3
    why = [g["why"] for g in genetics.genes_of(c, c.select("vpoDN")[0])]
    assert "fruitless-expressing" in why                               # no confidence grade in FlyWire's labels
    s = genetics.summary(c)
    rows = {g["key"]: g for g in s["expression"]}
    assert rows["fru"]["high"] is None and rows["female"]["n"] == 2 and "FlyWire" in s["source"]
    assert "♀" in genetics.genotype(c, c.select("vpoDN"))["tags"]


def test_the_female_fly_runs_at_the_papers_gain(female, conn):
    assert FlyBrain(female, backend="numpy").gain == DEFAULT_GAIN["female"] == 1.0
    assert FlyBrain(conn, backend="numpy").gain == DEFAULT_GAIN["male"] == 0.65
    assert FlyBrain(female, gain=0.5, backend="numpy").gain == 0.5


def test_experiments_skip_what_this_fly_does_not_have(female):
    b = FlyBrain(female, backend="numpy")
    exp = Experiment("Sugar", {"LB3b,LB3c": 150, "LgLG3": 80}, 300,
                     [R("MN9", "MN9", 5, 400), R("TTMn", "TTMn", 40, 100), R("GNG232", "G2N-1", 5, 400)])
    res = run_experiment(b, exp, seeds=(0, 1))
    assert res.missing == ["LgLG3", "TTMn"] and not res.na
    by = {r.label: r for r in res.readouts}
    assert by["TTMn"].ok is None and by["MN9"].ok and by["MN9"].hz > 5
    assert res.ok is True
    assert "not in this fly" in __import__("virtual_fly.experiments", fromlist=["x"]).format_result(res)
    # a lesion that cannot be made, or nothing to stimulate, or nothing to read: the experiment cannot be done
    for bad in (Experiment("lesion", {"LB3b,LB3c": 150}, 300, [R("MN9", "MN9", 5, 400)], silence=("pIP10",)),
                Experiment("no input", {"LgLG3": 80}, 300, [R("MN9", "MN9", 0, 5)]),
                Experiment("no readout", {"LB3b,LB3c": 150}, 300, [R("TTMn", "TTMn", 0, 5)]),
                Experiment("only the driven cells left", {"prefix:pC1_": 60}, 300,
                           [R("prefix:pC1_", "pC1", 40, 90), R("pIP10", "pIP10", 30, 150)])):
        r = run_experiment(b, bad)
        assert r.na and r.ok is None and not r.fragile and r.wall_s == 0.0


def test_load_connectome_female_builds_on_first_use(female, monkeypatch):
    from virtual_fly import connectome as C
    monkeypatch.setattr(flywire, "ensure_female", lambda quiet=False: female.path)
    monkeypatch.setattr(C, "download_connectome", lambda *a, **k: pytest.fail("tried to download the male file"))
    c = C.load_connectome(female=True, quiet=True)
    assert c.sex == "female" and c.select("MN9").size == 2
    with pytest.raises(ValueError):
        C.load_connectome(female.path, female=True)


def test_sources_are_pinned():
    for s in flywire.SOURCES.values():
        assert len(s["sha256"]) == 64 and "/raw.githubusercontent.com/" in s["url"]
        commit = s["url"].split("/")[5]
        assert len(commit) == 40 and all(ch in "0123456789abcdef" for ch in commit)
    assert set(flywire.ALIASES) >= {"MN9", "GNG232", "GNG087", "LB3b", "LB3c", "prefix:pC1_"}


def test_the_game_runs_on_a_female_fly(female):
    import json
    from virtual_fly.game import Game
    from virtual_fly.settings import build_brain
    g = Game(build_brain(female, "game", seed=0, backend="numpy"), autopilot=True, seed=1)
    lay = json.loads(g.layout_json)
    assert lay["sex"] == "female" and lay["columnar_vision"] is False     # no medulla columns in FlyWire's annotations
    keys = {r["key"] for r in lay["readouts"]}
    assert "MN9" in keys and "pIP10" not in keys and "TTMn" not in keys  # no gauge for cells this fly lacks
    for _ in range(5):
        g.tick()


def test_no_pyarrow_fails_before_downloading(tmp_path, monkeypatch):
    import builtins
    real = builtins.__import__

    def fake(name, *a, **k):
        if name.startswith("pyarrow"):
            raise ImportError(name)
        return real(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", fake)
    monkeypatch.setattr(flywire, "download_sources", lambda *a, **k: pytest.fail("downloaded without pyarrow"))
    with pytest.raises(SystemExit, match="pyarrow"):
        flywire.build_female(tmp_path / "f.flyb.gz", tmp_path, quiet=True)


def test_known_transmitters_follow_the_literature_column():
    ann = [_ann(1, "KCab", nt="dopamine", known="acetylcholine; sNPF; acetylcholine, sNPF"),
           _ann(2, "KCab", nt="dopamine", known="acetylcholine; sNPF"),
           _ann(3, "Delta7", nt="glutamate", known="glutamate, serotonin, proctolin, gaba-negative, dopamine-negative"),
           _ann(4, "Sm03", nt="glutamate", known="acetylcholine, nitric oxide, dopamine"),
           _ann(5, "Sm03", nt="glutamate"), _ann(6, "Sm03", nt="glutamate"),        # one labelled cell of three
           _ann(7, "NP", known="sNPF, gaba-negative")]                            # nothing the kit models
    k = flywire.known_transmitters(ann)
    assert k["KCab"]["nt"] == ["acetylcholine"] and "TAPIN" in k["KCab"]["source"]
    assert k["Delta7"]["nt"] == ["glutamate", "serotonin"]                      # negatives and peptides left out
    assert "Sm03" not in k and "NP" not in k


def test_a_low_confidence_prediction_is_unclear_but_keeps_its_sign(tmp_path):
    c = _female(tmp_path, more=[_ann(200, "LowConf", nt="glutamate", conf="0.31"), _ann(201, "HighConf", nt="glutamate")])
    lo, hi = c.select("LowConf")[0], c.select("HighConf")[0]
    assert c.nt[lo] == "unclear" and c.sign[lo] == -1                          # the published model's sign
    assert c.nt[hi] == "glutamate" and c.sign[hi] == -1


def test_the_parts_list_reads_flywires_literature(tmp_path):
    from virtual_fly import parts as P
    c = _female(tmp_path, more=[_ann(300, "KCab", nt="dopamine", known="acetylcholine; sNPF"),
                                _ann(301, "KCab", nt="dopamine", known="acetylcholine; sNPF"),
                                _ann(302, "DPM", nt="dopamine", known="serotonin; amnesiac; gaba; gaba, serotonin")])
    cp = P.PartsList().compile(c)
    kc, dpm = cp.role(c.select("KCab")[0]), cp.role(c.select("DPM")[0])
    assert kc["modulator"] is None and kc["sign"] == 1 and kc["curated"]["action"] == "not a modulator: fast synapses kept"
    assert dpm["modulator"] == "serotonin" and dpm["keep_fast"] and dpm["sign"] == -1   # as the male's curated DPM
    rows = {r["type"]: r for r in cp.counts["curated"]["rows"]}
    assert rows["KCab"]["source"].startswith("Davis") and rows["KCab"]["fbbt"] == ""    # no ontology class needed
    off = P.PartsList(curated="off").compile(c)                                  # the policy still decides
    assert off.role(c.select("KCab")[0])["modulator"] == "dopamine"


def test_an_old_female_file_is_rebuilt(tmp_path, monkeypatch):
    c = _female(tmp_path)                                                        # its meta has no build number
    assert flywire.built_with(c.path) == 0
    monkeypatch.setattr(flywire, "FEMALE_FILE", c.path)
    monkeypatch.setattr(flywire, "build_female", lambda quiet=False: "rebuilt")
    assert flywire.ensure_female() == "rebuilt"
    monkeypatch.setattr(flywire, "BUILD", 0)
    assert flywire.ensure_female() == c.path
