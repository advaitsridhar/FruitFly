"""The two builders of the Virtual Fly Brain data files (tools/): the OBO parser and type matcher, and the
harvest merge, on a tiny hand-written ontology and harvest."""

import gzip
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools import build_vfb_data as B  # noqa: E402
from tools import merge_vfb_harvest as M  # noqa: E402

OBO = r'''format-version: 1.2
data-version: fbbt/releases/2099-01-01

[Term]
id: FBbt:00005106
name: neuron

[Term]
id: FBbt:00047095
name: adult neuron
is_a: FBbt:00005106 ! neuron

[Term]
id: FBbt:00001446
name: larval neuron
is_a: FBbt:00005106 ! neuron

[Term]
id: FBbt:00048491
name: female-specific anatomical entity

[Term]
id: FBbt:00007173
name: cholinergic neuron
is_a: FBbt:00005106 ! neuron

[Term]
id: FBbt:00005131
name: dopaminergic neuron
is_a: FBbt:00005106 ! neuron

[Term]
id: FBbt:00003718
name: lamina monopolar neuron
def: "A neuron of the lamina." [FlyBase:FBrf0000001]
is_a: FBbt:00047095 ! adult neuron

[Term]
id: FBbt:00003719
name: lamina monopolar neuron L1
def: "Lamina monopolar neuron with a \"quoted\" word." [FlyBase:FBrf0000002]
synonym: "L1" EXACT VFB_SYMBOL [FlyBase:FBrf0000002]
is_a: FBbt:00003718 ! lamina monopolar neuron
is_a: FBbt:00007173 ! cholinergic neuron

[Term]
id: FBbt:00049318
name: adult lateral horn AD1a1 neuron
synonym: "L1" EXACT VFB_SYMBOL []
is_a: FBbt:00047095 ! adult neuron

[Term]
id: FBbt:00000010
name: larval thing L1
synonym: "L1" EXACT []
is_a: FBbt:00001446 ! larval neuron

[Term]
id: FBbt:00000020
name: adult ER3a neuron
synonym: "ER3a" EXACT VFB_SYMBOL []
is_a: FBbt:00047095 ! adult neuron
is_a: FBbt:00005131 ! dopaminergic neuron

[Term]
id: FBbt:00000030
name: adult female-only neuron
synonym: "vpoEN" EXACT VFB_SYMBOL []
is_a: FBbt:00047095 ! adult neuron
is_a: FBbt:00048491 ! female-specific anatomical entity

[Term]
id: FBbt:00000040
name: adult TmY21 neuron
synonym: "TmY21" EXACT VFB_SYMBOL []
synonym: "Tm36" RELATED []
is_a: FBbt:00047095 ! adult neuron

[Term]
id: FBbt:00000050
name: adult twin A neuron
synonym: "TWIN" EXACT []
is_a: FBbt:00047095 ! adult neuron

[Term]
id: FBbt:00000051
name: adult twin B neuron
synonym: "TWIN" EXACT []
is_a: FBbt:00047095 ! adult neuron

[Term]
id: FBbt:00000060
name: adult MNad04 neuron
synonym: "MNad04" EXACT VFB_SYMBOL []
is_a: FBbt:00047095 ! adult neuron

[Term]
id: FBbt:00000061
name: adult MNad48 neuron
synonym: "MNad48" EXACT VFB_SYMBOL []
is_a: FBbt:00047095 ! adult neuron

[Term]
id: FBbt:00000070
name: obsolete old neuron
synonym: "OLD" EXACT VFB_SYMBOL []
is_obsolete: true

[Term]
id: FBbt:00000080
name: adult SMP053 neuron
synonym: "SMP053" EXACT VFB_SYMBOL []
is_a: FBbt:00047095 ! adult neuron

[Term]
id: FBbt:00000081
name: adult CB0136 neuron
is_a: FBbt:00047095 ! adult neuron
is_a: FBbt:00007173 ! cholinergic neuron
'''

KIT = {"L1": 10, "ER3a_b": 3, "vpoEN": 2, "Tm36": 4, "TmY21": 5, "TWIN": 2, "MNad04,MNad48": 4, "OLD": 1, "SMP053": 2,
       "NOPE": 7}


@pytest.fixture(scope="module")
def obo(tmp_path_factory):
    p = tmp_path_factory.mktemp("obo") / "fbbt.obo"
    p.write_text(OBO)
    return p


def test_the_parser_and_the_strict_matcher(obo):
    t = B.clean(B.parse_obo(obo))
    assert t["FBbt:00003719"]["def"] == 'Lamina monopolar neuron with a "quoted" word.' and t["FBbt:00000070"]["obsolete"]
    fbbt_map, tree, unresolved = B.build(obo, KIT)
    assert fbbt_map["L1"]["fbbt"] == ["FBbt:00003719"] and fbbt_map["L1"]["route"] == "obo_symbol"   # label ends with the code
    assert fbbt_map["L1"]["nt"] == ["acetylcholine"] and fbbt_map["L1"]["evidence"] == "literature"
    assert fbbt_map["ER3a_b"] == {"fbbt": ["FBbt:00000020"], "route": "obo_stem", "n": 3, "coarse": True, "stem": "ER3a",
                                  "nt": ["dopamine"], "evidence": "literature"}
    assert fbbt_map["MNad04,MNad48"]["fbbt"] == ["FBbt:00000060", "FBbt:00000061"] and fbbt_map["MNad04,MNad48"]["route"] == "obo_split"
    assert fbbt_map["TmY21"]["fbbt"] == ["FBbt:00000040"]
    assert "vpoEN" in unresolved and "Tm36" in unresolved and "OLD" in unresolved and "NOPE" in unresolved   # female, RELATED, obsolete
    assert unresolved["TWIN"]["route"] == "ambiguous" and len(unresolved["TWIN"]["candidates"]) == 2
    assert "FBbt:00001446" not in tree["classes"] and tree["classes"]["FBbt:00003719"]["symbol"] == "L1"
    assert tree["source"]["release"] == "fbbt/releases/2099-01-01" and tree["classes"]["FBbt:00003718"]["parents"] == ["FBbt:00047095"]


def test_the_overlay_fills_corrects_and_rejects(obo):
    overlay = {"NOPE": {"fbbt": ["FBbt_00000081"], "route": "name_in_male-cns"},                  # a name the OBO lacks
               "SMP053": {"fbbt": ["FBbt_00000081"], "route": "name_in_male-cns", "override": True, "was": ["FBbt_00000080"]},
               "L1": {"fbbt": ["FBbt_00000081"], "route": "name_in_male-cns"},                    # no override: the OBO match stays
               "ER3a_b": {"fbbt": [], "route": "rejected", "was": ["FBbt:00000020"]},
               "TmY21": {"fbbt": [], "route": "rejected", "was": ["FBbt:00099999"]}}             # recorded against another class
    fbbt_map, _, unresolved = B.build(obo, KIT, overlay)
    assert fbbt_map["NOPE"]["fbbt"] == ["FBbt:00000081"] and fbbt_map["NOPE"]["route"] == "name_in_male-cns"
    assert fbbt_map["SMP053"]["fbbt"] == ["FBbt:00000081"] and fbbt_map["L1"]["fbbt"] == ["FBbt:00003719"]
    assert "ER3a_b" not in fbbt_map and unresolved["ER3a_b"]["route"] == "rejected"
    assert fbbt_map["TmY21"]["fbbt"] == ["FBbt:00000040"]


def test_the_command_line_writes_both_files(obo, tmp_path, capsys):
    types = tmp_path / "types.json"
    types.write_text(json.dumps(KIT))
    B.main(["--obo", str(obo), "--types", str(types), "--out", str(tmp_path / "data"), "--report", str(tmp_path / "u.json")])
    out = capsys.readouterr().out
    assert "mapped 5 of 10 types" in out and (tmp_path / "data" / "fbbt_map.json.gz").exists()
    m = json.load(gzip.open(tmp_path / "data" / "fbbt_map.json.gz", "rt"))
    assert set(m["types"]) == {"L1", "ER3a_b", "MNad04,MNad48", "TmY21", "SMP053"} and json.loads((tmp_path / "u.json").read_text())
    types.write_text("{}")
    B.main(["--obo", str(obo), "--types", str(types), "--out", str(tmp_path / "empty")])      # no division by zero
    assert "mapped 0 of 0 types" in capsys.readouterr().out


def _write(d: Path, name: str, obj):
    (d / name).write_text(obj if isinstance(obj, str) else json.dumps(obj))


def test_the_harvest_merge(tmp_path, capsys):
    h = tmp_path / "harvest"
    h.mkdir()
    _write(h, "overlay_0.json", [{"type": "R1-R6", "status": "resolved", "fbbt": ["FBbt_00006007"], "route": "name_in_male-cns",
                                  "label": "outer photoreceptor cell"},
                                 {"type": "TWO", "status": "ambiguous", "fbbt": ["FBbt_00000001", "FBbt_00000002"]},
                                 {"type": "BAD", "status": "resolved", "fbbt": ["FBbt_123"]}])
    _write(h, "overlay_0_progress.json", [{"type": "IGNORED", "status": "resolved", "fbbt": ["FBbt_00000009"]}])
    _write(h, "verify_0.json", [{"type": "SMP053", "verdict": "other_name", "fbbt": "FBbt_00000080"},
                                {"type": "PS008_b", "verdict": "not_this_class", "fbbt": ["FBbt_20001763"]}])
    _write(h, "corrections_checked.json", {"types": {"SMP053": {"fbbt": "FBbt_00000081", "was": "FBbt_00000080", "evidence": "checked"}}})
    _write(h, "routec_0.json", [{"type": "pC1_1a", "status": "found", "body_id": 1, "vfb_id": "VFB_x",
                                 "cell_type_classes": [{"fbbt": "FBbt_00110621", "label": "adult fruitless P1 (male) neuron"}]}])
    _write(h, "datasets.json", {"families": {"FCA_MALE": {"match": "FCA_MALE", "stage": "adult", "sex": "male", "name": "FCA male"},
                                             "AFCA": {"match": "AFCA", "stage": "adult", "sex": "mixed"},
                                             "FCA": {"match": "FCA", "stage": "adult", "sex": "mixed"},
                                             "KURM": {"match": "Kurmangaliyev", "stage": "pupal", "sex": "mixed"}}})
    rows = [{"fblc": "FBlc1", "cluster": "scRNAseq_2022_FCA_MALE_gamma KC", "anatomy_fbbt": "FBbt_00100247", "level": "002091.75", "extent": 0.8},
            {"fblc": "FBlc2", "cluster": "scRNAseq_2023_AFCA_gamma KC", "anatomy_fbbt": "FBbt_00100247", "level": 10.0, "extent": 0.5},
            {"fblc": "FBlc3", "cluster": "scRNAseq_2020_Kurmangaliyev_T4a", "anatomy_fbbt": "FBbt_00003732", "level": 1.0, "extent": 0.3}]
    _write(h, "rx_Dop1R1.json", {"gene": "Dop1R1", "fbgn": "FBgn0011582", "count": 3, "count_status": "exact", "coupling": "Gs", "rows": rows})
    _write(h, "rx_TyrR.json", {"gene": "TyrR", "count": 1, "count_status": "exact", "rows": rows[:1]})
    _write(h, "rx_Dop2R.json", {"gene": "Dop2R", "count": 688, "count_status": "exact", "rows": rows[:1]})        # incomplete
    _write(h, "rx_Oamb.json", '{"gene": "Oamb", "rows": [')                                                        # truncated
    ov = M.merge_overlay(h)
    assert ov["types"]["R1-R6"]["fbbt"] == ["FBbt:00006007"] and "IGNORED" not in ov["types"] and "BAD" not in ov["types"]
    assert "TWO" in ov["ambiguous"] and ov["types"]["pC1_1a"] == {"fbbt": ["FBbt:00110621"], "route": "vfb_individual",
                                                                  "label": "adult fruitless P1 (male) neuron",
                                                                  "source": "MaleCNS:1 -> VFB_x", "coarse": True}
    assert ov["types"]["SMP053"]["override"] and ov["types"]["SMP053"]["was"] == ["FBbt:00000080"]
    assert ov["types"]["PS008_b"] == {"fbbt": [], "route": "rejected", "was": ["FBbt:20001763"],
                                      "note": "the class carries MaleCNS names, none of them this one"}
    rx = M.merge_receptors(h)
    err = capsys.readouterr().err
    assert "rx_Dop2R.json: incomplete" in err and "rx_Oamb.json" in err
    assert set(rx["genes"]) == {"Dop1R1", "TyrR"} and rx["genes"]["Dop1R1"]["sign"] == 1 and rx["genes"]["TyrR"]["modulator"] is None
    assert rx["clusters"]["FBlc1"]["family"] == "FCA_MALE" and rx["clusters"]["FBlc2"]["family"] == "AFCA"          # not "FCA"
    assert rx["clusters"]["FBlc3"]["stage"] == "pupal" and rx["classes"]["FBbt:00100247"]["Dop1R1"][0] == ["FBlc1", 0.8, 2091.75]
    assert rx["families"] == ["FCA_MALE", "FCA", "AFCA"]                     # the preference order, adult only
    M.main(["--harvest", str(h), "--out", str(tmp_path / "out"), "--overlay", str(tmp_path / "new" / "dir" / "overlay.json")])
    assert (tmp_path / "new" / "dir" / "overlay.json").exists() and (tmp_path / "out" / "vfb_receptors.json.gz").exists()
