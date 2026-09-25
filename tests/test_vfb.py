"""What Virtual Fly Brain adds (vfb.py): the ontology selector, the popover facts, the receptor table,
the curated-transmitter overrides and the receptor signs in the parts list (synthetic connectome with
the hand-made ontology slice of ``synthetic_connectome.build_mini_vfb``)."""

import numpy as np
import pytest

from virtual_fly import vfb
from virtual_fly.brain import FlyBrain
from virtual_fly.parts import PartsList


def _set(conn, spec):
    return set(conn.select(spec).tolist())


def test_the_ontology_closure_and_the_fbbt_selector(conn, mini_vfb):
    ont, _ = mini_vfb
    assert not ont.empty and ont.resolve("FBbt_00003870") == ["FBbt:00003870"] == ont.resolve("lobula columnar neuron")
    assert ont.resolve("lc4") == ["FBbt:00003874"] and ont.resolve("nothing of the sort") == []
    assert ont.types_under("FBbt:00003870") == ["LC10a", "LC11", "LC4"]
    assert _set(conn, "fbbt:lobula columnar neuron") == _set(conn, "LC4,LC10a,LC11") == _set(conn, "fbbt:FBbt_00003870")
    assert _set(conn, "fbbt:lobula columnar neuron/L") == _set(conn, "LC4/L,LC10a/L,LC11/L")
    assert _set(conn, "fbbt:dopaminergic neuron") == _set(conn, "PPL101,PAM01")
    assert _set(conn, "fbbt:adult descending neuron&nt:acetylcholine") == _set(conn, "DNp01,DNa02,DNg74_a,DNg74_b") & _set(conn, "nt:acetylcholine")
    assert _set(conn, "fbbt:adult descending neuron") == _set(conn, "DNp01,DNa02,DNg74_a,DNg74_b")     # the coarse stem types count
    assert _set(conn, "fbbt:peptidergic neuron") == _set(conn, "KCg-m,KCab-m")
    assert _set(conn, "fbbt:adult SLPa&l1 lineage neuron") == _set(conn, "LC11") == _set(conn, "fbbt:FBbt_00050005")   # '&' in a label
    assert _set(conn, "fbbt:adult SLPa&l1 lineage neuron&nt:acetylcholine") == _set(conn, "LC11")
    assert conn.count("fbbt:neuron") == sum(conn.count(t) for t in ont.types)
    with pytest.raises(ValueError):
        conn.select("fbbt:no such class")
    with pytest.raises(ValueError):
        conn.select("fbbt:")


def test_describe_type_search_and_class_info(conn, mini_vfb):
    ont, _ = mini_vfb
    d = vfb.describe_type("LC4", conn)
    assert d["label"] == "lobula columnar neuron LC4" and d["url"].endswith("/reports/FBbt_00003874") and d["route"] == "obo_symbol"
    assert d["curated_nt"] == ["acetylcholine", "glutamate"] and d["evidence"] == "literature" and not d["coarse"]
    assert [b["label"] for b in d["breadcrumb"]] == ["lobula columnar neuron", "adult neuron", "neuron"]
    assert d["definition"].startswith("Lobula columnar neuron") and d["shared_by"] == 0
    kc = vfb.describe_type("KCg-m", conn)
    assert kc["peptides"] == ["sNPF"] and kc["breadcrumb"][0]["label"] == "adult Kenyon cell"     # not the transmitter parent
    assert ont.tags("FBbt:00003763")["peptides"] == ["Pdf"]         # 'l-LNv neuron' is anatomy below 'Pdf neuron', not a peptide
    assert vfb.describe_type("LC11", conn)["lineage"] == ["adult SLPa&l1 lineage neuron"]
    assert vfb.describe_type("KCab-m", conn)["coarse"] and vfb.describe_type("DNg74_a", conn)["shared_by"] == 2
    assert vfb.describe_type("DNp01", conn)["birth"] == "primary" and vfb.describe_type("MN9", conn) is None
    hits = ont.search("lobula", conn)
    assert hits[0]["label"] == "lobula columnar neuron" and hits[0]["types"] == 3 and hits[0]["neurons"] == conn.count("LC4,LC10a,LC11")
    assert hits[0]["spec"] == "fbbt:FBbt:00003870" and {h["label"] for h in hits[1:]} == {"lobula columnar neuron LC4", "lobula columnar neuron LC10a", "lobula columnar neuron LC11"}
    assert ont.search("", conn) == [] and ont.search("zzz", conn) == []
    info = ont.class_info("FBbt_00003870", conn)
    assert info["label"] == "lobula columnar neuron" and [k["label"] for k in info["children"]][0] in ("lobula columnar neuron LC4", "lobula columnar neuron LC10a")
    assert {t["type"] for t in info["types"]} == {"LC4", "LC10a", "LC11"} and info["n_types"] == 3 and info["parents"][0]["label"] == "adult neuron"
    assert ont.class_info("FBbt:99999999", conn) is None
    s = ont.summary(conn)
    assert s["available"] and s["types_mapped"] == len(ont.types) and s["curated"]["agree"] >= 6
    assert {r["type"] for r in s["curated"]["differ_rows"]} == {"GNG087", "LB1a"} and s["curated"]["unclear_with_curated"] == 1


def test_the_receptor_table_and_the_rx_selector(conn, mini_vfb):
    ont, rx = mini_vfb
    kc = rx.for_type("KCg-m", ont, conn)
    # 5-HT7 is listed by one of the two clusters: the other has it below 20 %, which counts as 0 in the mean
    assert kc["family"] == "FCA_MALE" and kc["depth"] == 0 and kc["n_clusters"] == 2
    assert kc["extent"] == {"Dop1R1": 0.8, "Dop2R": 0.75, "5-HT1A": 0.3, "5-HT7": 0.2}
    lc = rx.for_type("LC4", ont, conn)
    assert lc["class"] == "FBbt:00003870" and lc["depth"] == 1 and lc["family"] == "DAVIE"          # inherited from the parent class
    assert rx.for_type("DNa02", ont, conn) is None                                                # only a pupal cluster: ignored
    assert rx.for_type("MBON11", ont, conn) is None and rx.for_type("MN9", ont, conn) is None
    assert rx.sign(kc["extent"], "dopamine") == pytest.approx(0.05) and rx.sign(kc["extent"], "serotonin") == pytest.approx(-0.1)
    assert rx.sign({"Oamb": 0.3, "Octα2R": 0.6}, "octopamine") == pytest.approx(-0.3) and rx.sign({}, "dopamine") == 0.0
    assert _set(conn, "rx:Dop2R") == _set(conn, "KCg-m,LC4,LC10a,LC11") and _set(conn, "rx:dop2r>0.6") == _set(conn, "KCg-m")
    assert _set(conn, "rx:Dop2R>=0.75") == _set(conn, "KCg-m") and _set(conn, "rx:5-HT7") == _set(conn, "KCg-m")
    assert conn.select("rx:5-HT7>0.3").size == 0
    for bad in ("rx:Dop9R", "rx:Dop2R<0.5"):
        with pytest.raises(ValueError):
            conn.select(bad)
    r = vfb.receptors_of_type("KCg-m", conn)
    assert [x["gene"] for x in r["receptors"]] == ["Dop1R1", "Dop2R", "5-HT1A", "5-HT7"] and r["family_label"] == "Fly Cell Atlas (male)"
    # a partial harvest: a modulator none of whose receptors was harvested keeps the one-sign rule (None), not 0
    partial = vfb.Receptors({"genes": {g: v for g, v in rx.genes.items() if v["modulator"] != "serotonin"},
                             "clusters": rx.clusters, "classes": rx.classes, "families": rx.families})
    assert partial.sign(kc["extent"], "serotonin") is None and partial.sign(kc["extent"], "dopamine") == pytest.approx(0.05)
    not_harvested = vfb.Receptors({"genes": {g: {**v, "harvested": v["modulator"] != "octopamine"} for g, v in rx.genes.items()},
                                   "clusters": rx.clusters, "classes": rx.classes, "families": rx.families})
    assert not_harvested.sign({"Oamb": 0.9}, "octopamine") is None
    assert r["receptors"][1]["sign"] == -1 and r["receptors"][0]["flybase"].endswith("FBgn0011582")


def test_selections_follow_the_installed_data(conn, mini_vfb):
    ont, rx = mini_vfb
    assert conn.count("fbbt:lobula columnar neuron") == conn.count("LC4,LC10a,LC11")
    vfb.use(vfb.Ontology(), vfb.Receptors())
    try:
        with pytest.raises(ValueError):                     # not the cached answer from the other ontology
            conn.select("fbbt:lobula columnar neuron")
        with pytest.raises(ValueError, match="not installed"):
            conn.select("rx:Dop2R")
    finally:
        vfb.use(ont, rx)
    assert conn.count("fbbt:lobula columnar neuron") == conn.count("LC4,LC10a,LC11")


def _variant(mini, type_name, nts, evidence="literature", coarse=False):
    """The mini ontology with one type's curated transmitters replaced."""
    ont, _ = mini
    types = {t: dict(e) for t, e in ont.types.items()}
    e = dict(types.get(type_name, {"fbbt": ["FBbt:00090099"], "route": "obo_symbol", "n": 1}))
    e.update(nt=nts, evidence=evidence)
    if coarse:
        e["coarse"] = True
    types[type_name] = e
    return vfb.Ontology({"types": types}, {"classes": ont.classes, "roots": ont.roots, "source": ont.source})


# type, curated transmitters, policy -> action, modulator given, fast synapses kept, fast sign
BRANCHES = [
    ("LC4", ["acetylcholine", "dopamine"], "modulators", "co-release: fast synapses kept, tone added", "dopamine", True, 1),
    ("PPL101", ["dopamine", "gaba"], "modulators", "co-release: tone and fast synapses", None, True, -1),
    ("MBON20", ["glutamate", "octopamine"], "modulators", "unclear filled: fast synapses and a tone", "octopamine", True, -1),
    ("MBON20", ["serotonin"], "modulators", "unclear filled: a modulator", "serotonin", False, 1),
    ("MBON20", ["glutamate"], "modulators", "unclear filled: fast transmitter", None, False, -1),
    ("MBON20", ["acetylcholine", "dopamine", "gaba"], "modulators", "unclear filled: a tone, fast synapses left as predicted", "dopamine", True, 1),
    ("CSD", ["octopamine"], "modulators", "modulator changed", "octopamine", False, 1),
    ("CSD", ["gaba", "octopamine"], "modulators", "modulator changed", "octopamine", True, -1),
    ("CSD", ["acetylcholine"], "modulators", "not a modulator: fast synapses kept", "", False, 1),
    ("GNG087", ["acetylcholine", "octopamine"], "modulators", "tone added, predicted synapses kept", "octopamine", True, -1),
    ("GNG087", ["acetylcholine", "octopamine"], "all", "sign flipped and a tone added", "octopamine", True, 1),
    ("GNG087", ["glutamate", "octopamine"], "all", "tone added, predicted synapses kept", "octopamine", True, -1),   # both inhibitory: no flip
    ("GNG087", ["octopamine"], "modulators", "tone added (predicted synapses kept)", "octopamine", True, -1),
    ("GNG087", ["octopamine"], "all", "tone added, fast synapses removed", "octopamine", False, -1),
    ("GNG087", ["acetylcholine"], "all", "sign flipped", None, False, 1),
]


@pytest.mark.parametrize("type_name,nts,policy,action,mod,keep,sign", BRANCHES)
def test_every_override_branch(conn, mini_vfb, type_name, nts, policy, action, mod, keep, sign):
    vfb.use(_variant(mini_vfb, type_name, nts), mini_vfb[1])
    try:
        ov = vfb.transmitter_overrides(conn, policy)
    finally:
        vfb.use(*mini_vfb)
    idx = conn.select(type_name)
    assert {r["type"]: r["action"] for r in ov.rows}.get(type_name) == action
    assert all(ov.action[i] == action and ov.curated[i] == nts for i in idx.tolist())
    assert (ov.mod_nt[idx] == mod).all() and (ov.keep_fast[idx] == keep).all() and (ov.sign[idx] == sign).all()


def test_overrides_skip_what_should_only_annotate(conn, mini_vfb):
    for variant, t in ((_variant(mini_vfb, "GNG087", ["acetylcholine"], evidence="connectome"), "GNG087"),   # another data set
                       (_variant(mini_vfb, "GNG087", ["acetylcholine"], coarse=True), "GNG087"),            # a stem / individual
                       (_variant(mini_vfb, "LC4", ["acetylcholine"]), "LC4")):                              # agrees: nothing to do
        vfb.use(variant, mini_vfb[1])
        try:
            ov = vfb.transmitter_overrides(conn, "all")
        finally:
            vfb.use(*mini_vfb)
        assert t not in {r["type"] for r in ov.rows}
    # a curated modulator the parts list does not model is ignored (not reported as applied)
    vfb.use(_variant(mini_vfb, "LB1a", ["serotonin"]), mini_vfb[1])
    try:
        ov = vfb.transmitter_overrides(conn, "modulators", modulators=("dopamine", "octopamine"))
        cp = PartsList(modulators=PartsList().modulators[:2]).compile(conn)
    finally:
        vfb.use(*mini_vfb)
    assert "LB1a" not in {r["type"] for r in ov.rows} and not any(conn.types[i] == "LB1a" for i in cp.roles)


def test_curated_transmitter_policies(conn, mini_vfb):
    off = vfb.transmitter_overrides(conn, "off")
    assert off.counts["neurons"] == 0 and (off.sign == conn.sign).all() and not off.keep_fast.any()
    ov = vfb.transmitter_overrides(conn, "modulators")
    by = {r["type"]: r for r in ov.rows}
    # a predicted dopamine neuron the literature also calls GABAergic keeps fast (inhibitory) synapses
    pam = conn.select("PAM01")
    assert by["PAM01"]["action"].startswith("co-release") and ov.keep_fast[pam].all() and (ov.sign[pam] == -1).all() and (ov.mod_nt[pam] == None).all()  # noqa: E711
    # "unclear" filled from another connectome's class
    mb = conn.select("MBON20")
    assert by["MBON20"]["action"] == "unclear filled: fast transmitter" and (ov.sign[mb] == -1).all()
    # the literature calls a predicted cholinergic type octopaminergic: a tone is added, the synapses stay
    lb = conn.select("LB1a")
    assert (ov.mod_nt[lb] == "octopamine").all() and ov.keep_fast[lb].all() and (ov.sign[lb] == 1).all()
    # a confident fast prediction is not flipped under "modulators" ...
    assert "GNG087" not in by and (ov.sign[conn.select("GNG087")] == -1).all()
    assert ov.counts["policy"] == "modulators" and ov.counts["neurons"] == pam.size + mb.size + lb.size
    # ... but is under "all", and the octopaminergic type loses its fast synapses
    al = vfb.transmitter_overrides(conn, "all")
    by = {r["type"]: r for r in al.rows}
    assert by["GNG087"]["action"] == "sign flipped" and (al.sign[conn.select("GNG087")] == 1).all()
    assert not al.keep_fast[lb].any() and (al.mod_nt[lb] == "octopamine").all()
    with pytest.raises(ValueError):
        vfb.transmitter_overrides(conn, "sometimes")


def test_the_parts_list_applies_the_overrides_and_the_receptor_signs(conn, mini_vfb):
    cp = PartsList().compile(conn)
    lb, pam, oa = conn.select("LB1a"), conn.select("PAM01"), conn.select("OA-VPM3")
    assert (cp.mod_kind[lb] == 1).all() and cp.keep_fast[lb].all() and cp.keep_fast[pam].all() and not cp.keep_fast[oa].any()
    assert cp.counts["modulatory_neurons"] == 16 + lb.size and cp.counts["co_release_neurons"] == lb.size + pam.size
    assert cp.counts["curated"]["neurons"] == lb.size + pam.size + conn.count("MBON20") and len(cp.counts["curated"]["rows"]) == 3
    assert cp.role(int(lb[0])) == {"sign": 1, "modulator": "octopamine", "keep_fast": True, "graded": False, "theta_mv": 7.0,
                                   "curated": {"action": "tone added (predicted synapses kept)", "curated": ["octopamine"], "predicted": "acetylcholine"}}
    assert cp.role(int(conn.select("MN9")[0]))["modulator"] is None and "curated" not in cp.role(int(conn.select("MN9")[0]))
    # receptor signs: HSE's octopamine receptors are mostly Gi-coupled, so the tone lowers its gain
    hse = conn.select("HSE")
    pos = cp.target_pos[hse]
    assert (pos >= 0).all() and np.allclose(cp.mod_sign[1, pos], -0.3)
    assert np.allclose(cp.mod_sign[0, pos], 0.0)            # a cluster exists but expresses no dopamine receptor: no effect
    assert np.allclose(cp.mod_sign[0, cp.target_pos[conn.select("MBON11")]], 1.0)     # no cluster: the one-sign rule
    cov = {c["nt"]: c for c in cp.counts["receptor_signs"]["coverage"]}
    assert cov["octopamine"]["with_data"] >= hse.size and cov["octopamine"]["negative"] >= hse.size and cov["dopamine"]["targets"] == cp.mod_targets.size
    plain = PartsList(receptor_signs=False, curated="off").compile(conn)
    assert (plain.mod_sign == 1).all() and plain.counts["receptor_signs"]["coverage"] == [] and plain.counts["curated"]["neurons"] == 0
    assert (plain.sign == conn.sign).all() and not plain.keep_fast.any() and plain.counts["modulatory_neurons"] == 16


def _measure(brain, stim, ms=300, settle=100):
    brain.reset()
    brain.set_stimuli(stim)
    brain.run(settle)
    brain.reset_counts()
    brain.run(ms - settle)
    return brain


@pytest.mark.parametrize("backend", ["numpy", "numba"])
def test_the_brain_follows_the_overrides(conn, mini_vfb, backend):
    if backend == "numba":
        pytest.importorskip("numba")
    # a sound drives OA-VPM3; its octopamine tone now *lowers* the HS cells' gain (Gi receptors)
    signed = FlyBrain(conn, seed=0, parts=True, backend=backend)
    flat = FlyBrain(conn, seed=0, parts=PartsList(receptor_signs=False), backend=backend)
    for b in (signed, flat):
        _measure(b, {"prefix:JO-B": 120})
    hse = conn.select("HSE")
    g_signed, g_flat = signed._mod_gain[signed.parts.target_pos[hse]], flat._mod_gain[flat.parts.target_pos[hse]]
    assert (g_flat > 1.05).all() and (g_signed < 0.98).all() and (g_signed >= 0.1).all()
    assert signed.settings()["parts"]["receptor_signs"] and signed.settings()["parts"]["co_release_neurons"] == conn.count("LB1a,PAM01")
    # a co-releasing modulator keeps its fast synapses: PAM01 (now GABA + dopamine tone) still reaches MBON01
    co = FlyBrain(conn, seed=0, parts=True, backend=backend)
    cut = FlyBrain(conn, seed=0, parts=PartsList(curated="off"), backend=backend)
    assert (co.w[conn.out_edges(conn.select("PAM01"))] < 0).all() and (cut.w[conn.out_edges(conn.select("PAM01"))] == 0).all()
    # under "all" the bitter interneuron GNG087 is cholinergic, so bitter taste now drives MN9 instead of blocking it
    mod = FlyBrain(conn, seed=0, parts=PartsList(curated="modulators"), backend=backend)
    every = FlyBrain(conn, seed=0, parts=PartsList(curated="all"), backend=backend)
    assert (mod.w[conn.out_edges(conn.select("GNG087"))] < 0).all() and (every.w[conn.out_edges(conn.select("GNG087"))] > 0).all()
    assert _measure(mod, {"LB1a,LB1b": 150}).rate("MN9") == 0 and _measure(every, {"LB1a,LB1b": 150}).rate("MN9") > 20


def test_without_data_files_everything_degrades_to_nothing_known(conn, mini_vfb):
    vfb.use(vfb.Ontology(None, None), vfb.Receptors(None))
    conn._cache.clear()                                      # selections are cached per spec string
    try:
        assert vfb.ontology().empty and vfb.receptors().empty and vfb.describe_type("LC4", conn) is None
        assert vfb.receptors_of_type("LC4", conn) is None and vfb.ontology().summary(conn) == {"available": False}
        ov = vfb.transmitter_overrides(conn, "all")
        assert ov.counts["neurons"] == 0 and (ov.sign == conn.sign).all()
        sign, cov = vfb.receptor_signs(conn, np.arange(5), PartsList().modulators)
        assert sign.shape == (3, 5) and (sign == 1).all() and cov[0]["with_data"] == 0
        with pytest.raises(ValueError):
            conn.select("fbbt:lobula columnar neuron")
        with pytest.raises(ValueError):
            conn.select("rx:Dop2R")
    finally:
        vfb.use(*mini_vfb)
        conn._cache.clear()


def test_the_gain_follows_the_signed_tone_and_never_drops_below_the_floor(conn, mini_vfb):
    from virtual_fly.brain import GAIN_FLOOR
    b = FlyBrain(conn, seed=0, parts=True)
    k = b._mod_level.shape[0]
    b._mod_level[:] = 1e6                                   # every tone saturated on every target
    b._mod_sign = np.full_like(b._mod_sign, -1.0)           # and every receptor inhibitory: 1 - 0.3 - 0.5 - 0.3 < 0
    b._mod_block()
    assert k == 3 and np.allclose(b._mod_gain, GAIN_FLOOR) and b._mod_active
    b._mod_sign = np.full_like(b._mod_sign, 1.0)
    b._mod_level[:] = 1e6
    b._mod_block()
    assert np.allclose(b._mod_gain, 1 + 0.3 + 0.5 + 0.3, atol=1e-4)


def test_the_game_rebuilds_with_its_own_parts_list(conn, mini_vfb):
    from virtual_fly.game import Game
    from virtual_fly.settings import build_brain
    pl = PartsList(curated="all", receptor_signs=False)
    g = Game(build_brain(conn, "game", seed=0), autopilot=False, seed=1, parts_list=pl)
    assert not g.parts_on and g.parts_list() is pl and g.parts_arg(True) is pl and g.parts_arg(False) is False
    assert g.parts_counts()["curated"]["policy"] == "all" and b"\"curated\":\"all\"" in g._make_layout()
    plain = Game(build_brain(conn, "game", seed=0), autopilot=False, seed=1)
    assert plain.parts_arg(True) is True and plain.parts_list().curated == "modulators"
    on = Game(build_brain(conn, "game", seed=0, parts=pl), autopilot=False, seed=1)
    assert on.parts_on and on.parts_list() is pl and on.parts_arg(True) is pl
