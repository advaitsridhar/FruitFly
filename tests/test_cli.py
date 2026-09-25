"""The ``fly-brain`` command line, run in-process against the synthetic connectome.

``cli.main`` calls ``load_connectome()`` with no arguments, which would verify the checksum of the
default data file (and try to download it), so the loader is patched to return the synthetic
connectome instead. (Pointing ``FLY_DATA_FILE`` at a non-official file does not help: the checksum
check in ``download_connectome`` would still trigger a download; see the report.)
"""

import json

import numpy as np
import pytest

import virtual_fly.cli as cli
from virtual_fly.cli import main, parse_stim


@pytest.fixture(autouse=True)
def patched_loader(conn, monkeypatch):
    monkeypatch.setattr(cli, "load_connectome", lambda *a, **k: conn)


def test_parse_stim():
    assert parse_stim("MDN:60;LC4/R,LPLC2/R:150") == [("MDN", 60.0), ("LC4/R,LPLC2/R", 150.0)]
    assert parse_stim("MDN") == [("MDN", 80.0)]
    assert parse_stim(" MN9:12.5 ; ; class:DAN ") == [("MN9", 12.5), ("class:DAN", 80.0)]
    assert parse_stim("") == []


def test_find(capsys):
    main(["--find", "lc10"])
    out = capsys.readouterr().out
    assert "LC10a" in out and "20 neurons" in out and "1 types match 'lc10'" in out
    main(["--find", "zzz"])
    assert "0 types match" in capsys.readouterr().out


def test_info_and_bad_spec(capsys):
    main(["--info", "MN9"])
    out = capsys.readouterr().out
    assert out.count("type MN9") == 2
    with pytest.raises(SystemExit, match="No neurons match 'NOPE'.*--find NOPE"):
        main(["--info", "NOPE"])
    with pytest.raises(SystemExit, match="unknown filter"):
        main(["--info", "colour:red"])


def test_stim_and_watch_with_json(capsys, tmp_path):
    out_json = tmp_path / "rates.json"
    main(["--stim", "LB3b,LB3c:120", "--watch", "MN9;GNG232", "--ms", "300", "--json", str(out_json)])
    out = capsys.readouterr().out
    assert "stimulating LB3b,LB3c (20 neurons) at 120 Hz" in out and "simulated 300 ms" in out
    assert "MN9" in out and "GNG232" in out
    data = json.loads(out_json.read_text())
    assert data["stimulus"] == "LB3b,LB3c:120" and data["ms"] == 300
    assert data["rates"]["MN9"] > 20 and data["rates"]["GNG232"] > 20 and data["settings"]["dt"] == 0.5


def test_backend_flag_gives_the_same_rates(capsys):
    rates = {}
    for backend in ("numpy", "auto"):
        main(["--stim", "LB3b,LB3c:120", "--watch", "MN9", "--ms", "200", "--backend", backend])
        out = capsys.readouterr().out
        rates[backend] = [line for line in out.splitlines() if "MN9" in line]
    assert rates["numpy"] and rates["numpy"] == rates["auto"]          # identical spikes, whichever integrator


def test_parts_list_flags(capsys):
    main(["--parts", "--stim", "PPL101:300", "--watch", "MBON11;PPL101", "--ms", "200"])
    out = capsys.readouterr().out
    assert "parts list on: 16 modulatory neurons (dopamine, octopamine, serotonin)" in out and "graded cells" in out
    mbon = [l for l in out.splitlines() if l.strip().startswith("MBON11")][0]
    assert float(mbon.split()[-2]) == 0.0                                            # no fast dopamine synapses
    main(["--part", "GNG232:theta=3", "--stim", "LB3b:40", "--watch", "MN9", "--ms", "300"])
    out = capsys.readouterr().out
    assert "overrides: GNG232 -> theta_mv 3.0" in out
    assert float([l for l in out.splitlines() if l.strip().startswith("MN9")][0].split()[-2]) > 5
    with pytest.raises(SystemExit):
        main(["--part", "GNG232", "--stim", "LB3b:40"])


def test_curated_and_receptor_flags(capsys, mini_vfb):
    main(["--curated", "all", "--stim", "LB1a,LB1b:150", "--watch", "MN9", "--ms", "300"])
    out = capsys.readouterr().out
    assert "curated transmitters (all):" in out and "neurons in 4 types changed" in out and "receptor signs on" in out
    assert "APL releases locally (3 compartments, by lobe)" in out and "receptors from the literature for APL (Dop2R)" in out
    assert float([l for l in out.splitlines() if l.strip().startswith("MN9")][0].split()[-2]) > 20   # bitter now excites (test slice)
    main(["--parts", "--no-receptor-signs", "--curated", "off", "--stim", "LB1a,LB1b:150", "--watch", "MN9", "--ms", "300"])
    out = capsys.readouterr().out
    assert "curated transmitters" not in out and "receptor signs" not in out and "parts list on: 16 modulatory" in out
    assert float([l for l in out.splitlines() if l.strip().startswith("MN9")][0].split()[-2]) == 0.0
    main(["--global-apl", "--stim", "ORN_DM1:100", "--ms", "100"])
    out = capsys.readouterr().out
    assert "parts list on:" in out and "releases locally" not in out


def test_grow_and_genome_sweep(capsys):
    main(["--grow", "type", "--grow-seed", "2", "--stim", "LB3b,LB3c:120", "--watch", "MN9", "--ms", "200"])
    out = capsys.readouterr().out
    assert "grown a fly from its type wiring rules (seed 2)" in out and "MN9" in out
    main(["--genome-sweep", "real,type", "--profile", "game"])
    out = capsys.readouterr().out
    assert "real " in out and "type " in out and "experiments survive" in out and "experiment" in out


def test_stim_without_watch_lists_top_types(capsys):
    main(["--stim", "LB3b,LB3c:120", "--ms", "200", "--top", "3"])
    out = capsys.readouterr().out
    assert "most active cell types" in out and "GNG232" in out and "LB3b" not in out.split("most active")[1]


def test_stim_with_silence_modulate_record_and_profile(capsys, tmp_path):
    rec = tmp_path / "spikes.npz"
    out_json = tmp_path / "r.json"
    main(["--stim", "LB3b,LB3c:120", "--watch", "MN9", "--ms", "200", "--silence", "GNG232", "--modulate", "MN9:1.5",
          "--record", str(rec), "--profile", "game", "--json", str(out_json), "--seed", "2", "--fatigue", "0.1",
          "--noise", "0:1", "--std", "0.05:200", "--jitter", "0.5", "--gain", "0.6", "--kenyon-gain", "0.5"])
    out = capsys.readouterr().out
    assert "silencing GNG232: 4 neurons" in out and "modulating MN9 x1.5: 2 neurons" in out and "saved" in out
    data = json.loads(out_json.read_text())
    assert data["rates"]["MN9"] == 0                                          # the relay is silenced
    s = data["settings"]
    assert sorted(s["silenced"]) == ["GNG232", "class:ALLN", "class:DAN"] and s["modulated"] == {"MN9": 1.5}
    assert s["seed"] == 2 and s["fatigue_mv"] == 0.1 and s["std_u"] == 0.05 and s["std_tau_ms"] == 200
    assert s["threshold_jitter"] == 0.5 and s["gain"] == 0.6 and s["kenyon_gain"] == 0.5 and s["plasticity"]
    z = np.load(rec)
    assert set(z.files) == {"time_ms", "neuron", "body_id"} and z["neuron"].size > 0
    assert z["time_ms"].max() < 200 and np.array_equal(z["body_id"], 100000 + 7 * z["neuron"].astype(np.int64))


def test_trace(capsys, tmp_path):
    out_json = tmp_path / "trace.json"
    main(["--trace", "LC10a/L", "DNa02/L", "--json", str(out_json), "--hops", "3"])
    out = capsys.readouterr().out
    assert "strongest routes from LC10a/L to DNa02/L (up to 3 hops" in out
    assert "LC10a/L -(+)-> AOTU019/L" in out and "Relays carrying the most" in out and "AOTU019/L" in out
    data = json.loads(out_json.read_text())
    assert data[0]["nodes"] == ["LC10a/L", "AOTU019/L", "DNa02/L"]
    main(["--trace", "LC10a/L", "DNa02/L", "--avoid", "AOTU019"])
    assert "AOTU025/L" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        main(["--trace", "LC10a/L", "nowhere"])


def test_inputs_and_outputs(capsys):
    main(["--inputs", "MN9", "--outputs", "GNG232", "--top", "4"])
    out = capsys.readouterr().out
    assert "Strongest inputs of MN9 (2 neurons)" in out and "Strongest outputs of GNG232 (4 neurons)" in out
    assert "GNG232/L" in out and "gaba (-)" in out and "%" in out


def test_sweep(capsys, tmp_path):
    out_json = tmp_path / "sweep.json"
    main(["--sweep", "LB3b,LB3c:0:120:3", "--watch", "MN9,GNG232", "--ms", "300", "--json", str(out_json)])
    out = capsys.readouterr().out
    assert "Hz in" in out and "MN9" in out
    rows = json.loads(out_json.read_text())
    assert [r["hz"] for r in rows] == [0.0, 60.0, 120.0] and rows[0]["MN9"] == 0 and rows[-1]["MN9"] > 20
    with pytest.raises(SystemExit, match="--sweep needs --watch"):
        main(["--sweep", "LB3b,LB3c:0:120:3"])


def test_lesion(capsys, tmp_path):
    out_json = tmp_path / "lesion.json"
    main(["--lesion", "Sugar on the", "--readout", "MN9", "--candidates", "GNG232,DNp01", "--json", str(out_json)])
    out = capsys.readouterr().out
    assert "Sugar on the mouthparts: MN9 with each population silenced" in out
    rows = json.loads(out_json.read_text())
    assert rows[0]["silenced"] == "GNG232" and rows[0]["hz"] == 0 and {r["silenced"] for r in rows} == {"GNG232", "DNp01"}
    with pytest.raises(SystemExit, match="no experiment matches"):
        main(["--lesion", "unicorn"])


def test_run_experiments_with_only_and_seeds(capsys, tmp_path):
    out_json = tmp_path / "results.json"
    main(["--only", "silence", "--seeds", "2", "--json", str(out_json)])
    out = capsys.readouterr().out
    assert "Running the validated experiments with the 'pure' profile" in out
    assert "1/1 readouts in the expected range (1/1 experiments)" in out and "wrote" in out
    assert "Try the game's settings" in out
    data = json.loads(out_json.read_text())
    assert [r["name"] for r in data["results"]] == ["Silence (no input)"] and data["results"][0]["seeds"] == [0, 1]
    assert data["results"][0]["ok"] and data["settings"]["seed"] == 0
    main(["--only", "silence", "--profile", "game"])
    assert "Try the game's settings" not in capsys.readouterr().out
    main(["--only", "silence", "--json", str(out_json)])                      # five seeds unless told otherwise
    assert json.loads(out_json.read_text())["results"][0]["seeds"] == [0, 1, 2, 3, 4]


def test_module_entry_points_import():
    import virtual_fly
    from virtual_fly import play
    assert virtual_fly.__version__ and callable(play.main) and callable(cli.main)
