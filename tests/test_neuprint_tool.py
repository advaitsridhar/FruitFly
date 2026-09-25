"""tools/harvest_neuprint_rois.py without a network or a token: dataset choice, packing and the log summary."""
import pytest

from tools.harvest_neuprint_rois import pack, pick_dataset, summary


def test_pick_dataset():
    assert pick_dataset(["hemibrain:v1.2.1", "male-cns:v1.0", "male-cns:v1.1"], None) == "male-cns:v1.0"
    assert pick_dataset(["male-cns:v0.9", "male-cns:v1.1", "manc:v1.2"], None) == "male-cns:v1.1"
    assert pick_dataset(["male-cns:v1.1"], "male-cns:v1.1") == "male-cns:v1.1"
    with pytest.raises(SystemExit):
        pick_dataset(["hemibrain:v1.2.1"], None)
    with pytest.raises(SystemExit):
        pick_dataset(["male-cns:v1.0"], "male-cns:v9")


def test_pack_dedupes_and_summarises():
    neurons = [(1, "APL", "APL(R)"), (2, "KCg-m", "KCg-m(R)"), (3, "DPM", "DPM(R)"), (2, "KCg-m", "KCg-m(R)")]
    conns = [(2, 1, "gL(R)", 12), (2, 1, "CA(R)", 3), (1, 2, "CA(R)", 7), (1, 3, "gL(R)", 4),
             (1, 3, "gL(R)", 4),                                   # APL -> DPM comes back from both queries
             (3, 1, "NotPrimary", 0)]                              # an empty region row is dropped
    t = pack(neurons, conns, "male-cns:v1.0", "0.6.3")
    assert t["dataset"] == "male-cns:v1.0" and t["cells"] == ["APL", "DPM"] and set(t["rois"]) == {"gL(R)", "CA(R)"}
    assert len(t["edges"]) == 4 and t["neurons"]["1"] == ["APL", "APL(R)"]
    roi = {r: k for k, r in enumerate(t["rois"])}
    assert [2, 1, roi["gL(R)"], 12] in t["edges"] and [1, 3, roi["gL(R)"], 4] in t["edges"]
    text = summary(t)
    assert "APL inputs: 15 synapses; gL(R) 12, CA(R) 3" in text and "APL outputs: 11 synapses" in text
    assert "DPM inputs: 4 synapses" in text
