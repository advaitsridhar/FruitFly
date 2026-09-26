"""Gene-expression populations, the genotype of readouts, the NeuronBridge client (offline fake) and
silencing experiments, on the synthetic connectome."""

import json
import urllib.error

import numpy as np
import pytest

from virtual_fly import genetics as G
from virtual_fly.experiments import Experiment, R, run_experiment


def test_summary_counts_tags_and_links(conn):
    g = G.summary(conn, [{"key": "pIP10", "spec": "pIP10"}, {"key": "MN9", "spec": "MN9"}])
    exp = {e["key"]: e for e in g["expression"]}
    assert exp["fru"]["n"] == 30 and exp["fru"]["high"] == 10 and exp["fru"]["flybase"].endswith("FBgn0004652")
    assert exp["dsx"]["n"] == 10 and exp["dsx"]["high"] == 10 and exp["both"]["n"] == 8 and exp["both"]["flybase"] is None
    assert exp["male"]["n"] == 10 and exp["dimorphic"]["n"] == 2 and exp["fru"]["types"] == 3
    nts = {t["nt"]: t for t in g["transmitters"]}
    assert nts["gaba"]["sign"] == -1 and nts["acetylcholine"]["sign"] == 1 and nts["dopamine"]["n"] == conn.select("nt:dopamine").size
    assert 0.9 < sum(t["synapse_share"] for t in g["transmitters"]) <= 1.0
    assert [x["symbol"] for x in nts["acetylcholine"]["genes"]] == ["ChAT", "VAChT"]
    assert g["readouts"]["pIP10"]["tags"] == ["fru", "♂"] and g["readouts"]["MN9"]["tags"] == []
    assert {x["symbol"] for x in g["genes"]} >= {"fru", "dsx", "ChAT", "VGlut", "Gad1", "ple", "Tdc2"}
    assert all(x["flybase"].startswith("https://flybase.org/reports/FBgn") for x in g["genes"])
    assert "MaleCNS" in g["source"] and isinstance(g["unclear"], int)
    assert g["unclear"] == conn.select("nt:unclear").size > 0 and g["unclear_inhibitory"] == 0     # MaleCNS: all excitatory


def test_genotype_and_genes_of(conn):
    gt = G.genotype(conn, conn.select("pC1_1a"))
    assert gt["n"] == 8 and gt["fru"] == 1.0 and gt["dsx"] == 1.0 and gt["male"] == 1.0 and gt["dimorphic"] == 0.0
    assert gt["tags"] == ["fru", "dsx", "♂"] and gt["nt"] == "acetylcholine"
    assert G.genotype(conn, np.zeros(0, dtype=np.int64))["tags"] == []
    mixed = G.genotype(conn, conn.select("pC1_1a,MN9,GNG232,LB3b"))                # mostly not fru: no tag
    assert 0 < mixed["fru"] < 0.5 and mixed["tags"] == []
    genes = G.genes_of(conn, int(conn.select("pIP10/L")[0]))
    assert [x["symbol"] for x in genes] == ["fru", "ChAT", "VAChT"] and "high confidence" in genes[0]["why"]
    assert [x["symbol"] for x in G.genes_of(conn, int(conn.select("MN9")[0]))] == ["ChAT", "VAChT"]
    both = G.genes_of(conn, int(conn.select("pC1_1a")[0]))
    assert [x["symbol"] for x in both][:2] == ["fru", "dsx"]
    gaba = G.genes_of(conn, int(conn.select("GNG087")[0]))
    assert [x["symbol"] for x in gaba] == ["Gad1", "VGAT"]


# ------------------------------------------------------------------ NeuronBridge, offline
def make_fake(conn):
    """A tiny NeuronBridge: one pIP10 body with two matching lines; line SS00001 matches three bodies."""
    body = int(conn.body_id[conn.select("pIP10/L")[0]])
    other = int(conn.body_id[conn.select("MN9/L")[0]])
    V = "v9_9_9"
    mc = "FlyEM_Male_CNS_Brain_v0.9"
    files = {
        "/current.txt": V,
        f"/{V}/metadata/by_body/{body}.json": {"results": [
            {"libraryName": mc, "publishedName": f"male-cns:v0.9:{body}", "neuronType": "pIP10", "files": {"CDSResults": "em1.json"}},
            {"libraryName": "FlyEM_Hemibrain_v1.2.1", "publishedName": f"hemibrain:v1.2.1:{body}", "files": {"CDSResults": "hb.json"}}]},
        f"/{V}/metadata/cdsresults/em1.json": {"results": [
            {"normalizedScore": 900.0, "image": {"publishedName": "SS00001", "libraryName": "FlyLight Split-GAL4 Drivers"}},
            {"normalizedScore": 500.0, "image": {"publishedName": "R00A00", "libraryName": "FlyLight Gen1 MCFO v1.1"}},
            {"normalizedScore": 950.0, "image": {"publishedName": "SS00001", "libraryName": "FlyLight Split-GAL4 Drivers"}}]},
        f"/{V}/metadata/by_line/SS00001.json": {"results": [
            {"libraryName": "FlyLight Split-GAL4 Drivers", "files": {"CDSResults": "lm1.json"}},
            {"libraryName": "FlyLight Split-GAL4 Drivers", "files": {}}]},
        f"/{V}/metadata/cdsresults/lm1.json": {"results": [
            {"normalizedScore": 950.0, "image": {"libraryName": mc, "publishedName": f"male-cns:v0.9:{body}", "neuronType": "pIP10"}},
            {"normalizedScore": 700.0, "image": {"libraryName": mc, "publishedName": "male-cns:v0.9:999999999", "neuronType": "ghost"}},
            {"normalizedScore": 600.0, "image": {"libraryName": "FlyWire_FAFB_v783_realign", "publishedName": "fw:1"}},
            {"normalizedScore": 650.0, "image": {"libraryName": "FlyEM_Male_CNS_VNC_v0.9", "publishedName": f"male-cns:v0.9:{other}", "neuronType": "MN9-ish"}}]},
    }
    calls = []

    def fetch(url, timeout):
        calls.append(url)
        path = url[len(G.NB_BASE):]
        if path not in files:
            raise urllib.error.HTTPError(url, 404, "Not Found", None, None)
        v = files[path]
        return (v if isinstance(v, str) else json.dumps(v)).encode()
    return fetch, calls, body, other


def test_neuronbridge_lines_for_and_the_disk_cache(conn, tmp_path):
    fetch, calls, body, other = make_fake(conn)
    nb = G.NeuronBridge(cache_dir=tmp_path, fetch=fetch)
    r = nb.lines_for(conn, "pIP10/L")
    assert r["version"] == "v9_9_9" and r["n"] == 1 and r["sampled"] == [body] and r["unmatched"] == []
    assert [l["line"] for l in r["lines"]] == ["SS00001", "R00A00"]
    assert r["lines"][0]["score"] == 950.0 and r["lines"][0]["split"] and r["lines"][0]["neurons"] == 1 and not r["lines"][1]["split"]
    n = len(calls)
    nb2 = G.NeuronBridge(cache_dir=tmp_path, fetch=fetch)                        # a fresh client reads the disk cache
    assert nb2.lines_for(conn, "pIP10/L")["lines"][0]["line"] == "SS00001"
    assert len(calls) == n + 1                                                    # only current.txt is fetched again
    r = nb.lines_for(conn, "pIP10/L,MN9/L")                                       # a body NeuronBridge does not know
    assert r["sampled"] == sorted([body, other]) or set(r["sampled"]) == {body, other}
    assert r["unmatched"] == [other] and [l["line"] for l in r["lines"]] == ["SS00001", "R00A00"]
    with pytest.raises(ValueError, match="no neurons"):
        nb.lines_for(conn, "NOPE")
    big = nb.lines_for(conn, "class:Kenyon_Cell", max_neurons=3)                 # sampled, not all 60
    assert len(big["sampled"]) == 3 and big["lines"] == [] and len(big["unmatched"]) == 3


def test_neuronbridge_neurons_for_line(conn, tmp_path):
    fetch, calls, body, other = make_fake(conn)
    nb = G.NeuronBridge(cache_dir=tmp_path, fetch=fetch)
    r = nb.neurons_for_line(conn, "SS00001")
    assert r["line"] == "SS00001" and r["images"] == 1 and r["searched"] == 1 and "Split-GAL4" in r["library"]
    assert [n["body"] for n in r["neurons"]] == [body, 999999999, other]           # by score; other datasets dropped
    first, ghost, third = r["neurons"]
    assert first["in_kit"] and first["type"] == "pIP10" and first["side"] == "L" and first["nb_type"] == "pIP10"
    assert not ghost["in_kit"] and ghost["nb_type"] == "ghost" and ghost["index"] is None
    assert third["in_kit"] and third["type"] == "MN9" and third["nb_type"] == "MN9-ish"    # a disagreement is visible
    assert r["spec"] == f"body:{body},body:{other}" and conn.select(r["spec"]).size == 2
    with pytest.raises(ValueError, match="no line called"):
        nb.neurons_for_line(conn, "SS99999")
    with pytest.raises(ValueError, match="looks like"):
        nb.neurons_for_line(conn, "bad name!")


def test_neuronbridge_offline_is_one_clear_error(conn, tmp_path):
    def down(url, timeout):
        raise OSError("no network")
    nb = G.NeuronBridge(cache_dir=tmp_path, fetch=down)
    with pytest.raises(G.NeuronBridgeError, match="internet"):
        nb.lines_for(conn, "pIP10")
    with pytest.raises(G.NeuronBridgeError, match="internet"):
        nb.neurons_for_line(conn, "SS00001")
    assert not any(tmp_path.iterdir())                                            # nothing half-written


# ------------------------------------------------------------------ a genetic lesion in an experiment
def test_silencing_experiment_blocks_the_song_and_is_undone(brain):
    song = Experiment("song", {"pC1_1a": 80}, 300, [R("pIP10", "pIP10", 5, 500)])
    intact = run_experiment(brain, song)
    assert intact.ok and intact.silenced == [] and intact.to_dict()["silenced"] == []
    lesion = Experiment("song, fruitless silenced", {"pC1_1a": 80}, 300,                  # pC1_1a itself is fru+ here
                        [R("pIP10", "pIP10", 0, 0.0), R("MNwm35", "wing motor neurons", 0, 0.0)], silence=("gene:fru",))
    res = run_experiment(brain, lesion)
    assert res.ok and res.silenced == ["gene:fru"] and res.to_dict()["silenced"] == ["gene:fru"]
    assert brain.silenced == {}                                                   # the lesion is undone afterwards
    assert run_experiment(brain, song).ok                                         # and the fly sings again
