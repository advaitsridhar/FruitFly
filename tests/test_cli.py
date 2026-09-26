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
    assert "overrides: GNG232 (4 neurons) -> theta_mv 3.0" in out
    assert float([l for l in out.splitlines() if l.strip().startswith("MN9")][0].split()[-2]) > 5
    with pytest.raises(SystemExit):
        main(["--part", "GNG232", "--stim", "LB3b:40"])
    with pytest.raises(SystemExit, match="No neurons match 'NOPE'"):             # an override that would change nothing
        main(["--part", "NOPE:theta=10", "--stim", "LB3b:40"])
    with pytest.raises(SystemExit, match="unknown filter 'foo:'"):              # no traceback from a malformed spec
        main(["--part", "foo:bar:theta=10", "--stim", "LB3b:40"])


def test_curated_and_receptor_flags(capsys, mini_vfb, conn):
    main(["--curated", "all", "--stim", "LB1a,LB1b:150", "--watch", "MN9", "--ms", "300"])
    out = capsys.readouterr().out
    assert "curated transmitters (all):" in out and "neurons in 4 types changed" in out and "receptor signs on" in out
    assert "APL releases locally (3 compartments, by lobe)" in out and "receptors from the literature for APL (Dop2R)" in out
    from virtual_fly.parts import PartsList
    c = PartsList(curated="all").compile(conn).counts
    rs = c["receptor_signs"]
    unsigned = c["modulated_targets"] - max(r["with_data"] for r in rs["coverage"]) - sum(f["targets"] for f in rs["facts"])
    assert f"no tone on the {unsigned:,} targets without receptor data" in out     # a fact's targets have a tone
    assert float([l for l in out.splitlines() if l.strip().startswith("MN9")][0].split()[-2]) > 5   # bitter now excites (test slice; 0 under "modulators")
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
    assert "skipped" not in out
    main(["--genome-sweep", "real", "--seeds", "1"])                          # the pure profile says what it left out
    out = capsys.readouterr().out
    assert "/ 6 experiments survive" in out
    assert "(skipped 5 experiments whose ranges were measured with the 'game' profile: Smell of vinegar," in out
    assert "run them with --profile game)" in out


def test_stim_without_watch_lists_top_types(capsys, tmp_path):
    out_json = tmp_path / "top.json"
    main(["--stim", "LB3b,LB3c:120", "--ms", "200", "--top", "3", "--json", str(out_json)])
    out = capsys.readouterr().out
    assert "most active cell types" in out and "GNG232" in out and "LB3b" not in out.split("most active")[1]
    data = json.loads(out_json.read_text())                                   # the printed rows are saved too
    assert data["rates"] == {} and 0 < len(data["top_types"]) <= 3 and "GNG232" in {r["type"] for r in data["top_types"]}
    assert f"wrote {out_json}" in out


def test_a_misspelt_watch_fails_before_simulating(capsys):
    with pytest.raises(SystemExit, match="No neurons match 'MNN9'.*--find MNN9"):   # it used to read 0 Hz
        main(["--sweep", "LB3b,LB3c:0:120:3", "--watch", "MN9,MNN9"])
    with pytest.raises(SystemExit, match="No neurons match 'NOPE'"):
        main(["--stim", "LB3b,LB3c:120", "--watch", "MN9;NOPE"])
    assert "simulated" not in capsys.readouterr().out


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


def test_inputs_and_outputs(capsys, tmp_path):
    out_json = tmp_path / "partners.json"
    main(["--inputs", "MN9", "--outputs", "GNG232", "--top", "4", "--json", str(out_json)])
    out = capsys.readouterr().out
    assert "Strongest inputs of MN9 (2 neurons)" in out and "Strongest outputs of GNG232 (4 neurons)" in out
    assert "GNG232/L" in out and "gaba (-)" in out and "%" in out
    # the share is of the partner's own traffic, and the header says which
    assert out.split("Strongest outputs")[0].count("% of its output") == 1 and "% of its input" in out.split("Strongest outputs")[1]
    data = json.loads(out_json.read_text())
    assert data["inputs"]["spec"] == "MN9" and data["outputs"]["neurons"] == 4 and len(data["inputs"]["rows"]) <= 4
    assert {"type", "side", "synapses", "fraction"} <= set(data["outputs"]["rows"][0])


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
    assert f"(baseline {rows[0]['baseline']:.1f} Hz)" in out
    with pytest.raises(SystemExit, match="no experiment matches"):
        main(["--lesion", "unicorn"])
    with pytest.raises(SystemExit, match="No neurons match 'NOPE'"):             # a candidate that matches nothing
        main(["--lesion", "Sugar on the", "--readout", "MN9", "--candidates", "GNG232,NOPE"])


def test_lesion_finds_its_own_candidates_or_says_why_not(capsys):
    main(["--lesion", "Sugar on the", "--readout", "MN9"])
    out = capsys.readouterr().out
    line = next(l for l in out.splitlines() if l.startswith("lesion candidates"))
    names = line.split(": ", 1)[1].split(", ")
    assert "GNG232/L" in names and not any(n.startswith(("LB3", "PhG1", "LgLG3")) for n in names)   # no stimulated cells
    table = out.split("with each population silenced")[1].splitlines()[2:]
    assert len(table) == len(names) and all(l.split()[0] in names for l in table)
    with pytest.raises(SystemExit, match="no wiring route from the stimulus of 'Silence"):   # never an empty table
        main(["--lesion", "Silence"])


@pytest.mark.parametrize("argv, message", [
    (["--lesion", "Sugar on the", "--readout", "MNN9"], "--readout: 'MNN9' is not a readout of 'Sugar on the mouthparts'; "
                                                        "choose one of: GNG232, DNg67, MN9, class:Kenyon_Cell"),
    (["--lesion", "Sugar on the", "--readout", "GNG087", "--candidates", "GNG232"], "'GNG087' is not a readout"),
])
def test_lesion_readout_must_be_the_experiments(capsys, argv, message):
    with pytest.raises(SystemExit) as e:
        main(argv)
    err = capsys.readouterr().err
    assert e.value.code == 2 and message in err and err.count("\n") == 1


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
    assert data["results"][0]["fragile"] is False and "after_events_per_s" in data["results"][0]
    main(["--only", "silence", "--profile", "game"])
    assert "Try the game's settings" not in capsys.readouterr().out
    main(["--only", "silence", "--json", str(out_json)])                      # five seeds unless told otherwise
    assert json.loads(out_json.read_text())["results"][0]["seeds"] == [0, 1, 2, 3, 4]


def test_module_entry_points_import():
    import virtual_fly
    from virtual_fly import play
    assert virtual_fly.__version__ and callable(play.main) and callable(cli.main)


def test_one_sign_rule_flag(capsys, mini_vfb, monkeypatch):
    from virtual_fly import parts as P
    seen = []
    real = P.PartsList.compile
    monkeypatch.setattr(P.PartsList, "compile", lambda self, c: (seen.append(self.unknown_sign), real(self, c))[1])
    main(["--one-sign-rule", "--stim", "LB1a,LB1b:150", "--watch", "MN9", "--ms", "100"])
    out = capsys.readouterr().out
    assert "parts list on:" in out and "the one-sign rule for targets without receptor data (v2.7)" in out
    assert "no tone on the" not in out and seen and set(seen) == {1.0}


def test_the_summary_names_what_cannot_be_done_and_the_runaways(capsys, monkeypatch):
    from virtual_fly import experiments as E
    na = E.ExperimentResult("A", [E.ReadoutResult("x", "pIP10", 0.0, 0.0, 0, 3, None)], 0.0, "", 0.0, [0], na=True, missing=["pIP10"])
    ok = E.ExperimentResult("B", [E.ReadoutResult("y", "MN9", 30.0, 0.0, 20, 40, True, [30.0])], 60000.0,
                            E.after_note(60000.0), 0.1, [0], after_per_seed=[60000.0])
    monkeypatch.setattr(E, "run_all", lambda *a, **k: [na, ok])
    main([])
    out = capsys.readouterr().out
    assert "1 cannot be done on this fly: A" in out and "After the stimulus, 1 of 1 leave a runaway loop on at least one seed." in out
    assert "(+N) after a range" not in out                                    # no pass needed the margin


def test_a_pass_through_the_margin_is_shown(capsys, monkeypatch):
    from virtual_fly import experiments as E
    r = E.ExperimentResult("Sugar + bitter together", [E.ReadoutResult("MN9", "MN9", 5.8, 1.6, 0, 5, True, [4.0, 4.0, 6.0, 8.0, 7.0],
                                                                        seeds_out=2)], 0.0, "calm", 0.1, [0, 1, 2, 3, 4])
    monkeypatch.setattr(E, "run_all", lambda *a, **k: [r])
    main([])
    out = capsys.readouterr().out
    assert "(+N) after a range: a rate passes up to max(1 Hz, 15 %) above the top of its range" in out


@pytest.mark.parametrize("argv, message", [
    (["--sweep", "LB3b,LB3c:0:200", "--watch", "MN9"], "--sweep wants SPEC:LO:HI:N"),
    (["--sweep", "LB3b,LB3c:0:200:nine", "--watch", "MN9"], "--sweep wants SPEC:LO:HI:N"),
    (["--sweep", "LB3b,LB3c:0:200:0", "--watch", "MN9"], "--sweep wants SPEC:LO:HI:N"),
    (["--noise", "2", "--stim", "MN9:60"], "--noise wants HZ:MV"),
    (["--std", "0.1", "--stim", "MN9:60"], "--std wants U:TAU_MS"),
    (["--modulate", "LB3b,LB3c", "--stim", "LB3b:120"], "--modulate wants SPEC:FACTOR"),
    (["--grow", "typo"], "--grow: 'typo': level must be real, type, class or bottleneck:K (K = 1 to 2048)"),
    (["--grow", "bottleneck:abc"], "K = 1 to 2048"),
    (["--grow", "bottleneck:0"], "K = 1 to 2048"),
    (["--grow", "bottleneck:999999"], "K = 1 to 2048"),
    (["--genome-sweep", "real,bottleneck:-3"], "--genome-sweep: 'bottleneck:-3'"),
    (["--genome-sweep", ","], "--genome-sweep wants comma-separated levels"),
    (["--seeds", "0"], "--seeds must be at least 1"),
    (["--record", "x.npz"], "--record FILE.npz saves the spikes of a --stim run"),
    (["--only", "vinegr"], "no experiment's name or tag contains 'vinegr'; the tags are classic, courtship, escape"),
])
def test_malformed_options_stop_with_one_line_before_loading(capsys, monkeypatch, argv, message):
    monkeypatch.setattr(cli, "load_connectome", lambda *a, **k: pytest.fail("loaded the connectome"))
    with pytest.raises(SystemExit) as e:
        main(argv)
    err = capsys.readouterr().err
    assert e.value.code == 2 and err.count("\n") == 1 and message in err and "Traceback" not in err


def test_number_selectors_say_what_they_need():
    with pytest.raises(SystemExit) as e:
        main(["--info", "body:abc"])
    assert "body: needs a neuron's id, a whole number" in str(e.value) and "--find" not in str(e.value)
    with pytest.raises(SystemExit, match="index:999999999 is out of range"):
        main(["--info", "index:999999999"])
    with pytest.raises(SystemExit) as e:                                        # re.error, not a traceback
        main(["--stim", "regex:[:60"])
    assert "regex:[ is not a regular expression Python can read (unterminated" in str(e.value) and "--find" not in str(e.value)


def test_a_stimulus_without_a_rate_says_so(capsys):
    with pytest.raises(SystemExit) as e:
        main(["--stim", "MN9:abc"])
    err = capsys.readouterr().err
    assert e.value.code == 2 and "--stim: 'abc' is not a rate in Hz; use SPEC:HZ" in err
    main(["--stim", "class:Kenyon_Cell", "--ms", "50", "--watch", "MN9"])      # a filter with no rate still works (80 Hz)
    assert "stimulating class:Kenyon_Cell" in capsys.readouterr().out


@pytest.fixture
def female(conn, monkeypatch):
    """The synthetic fly relabelled as a FlyWire female: an alias, and two 'unclear' neurons that inhibit."""
    f = conn.rewired(conn.row_ptr, conn.post_idx, conn.n_syn, label="female")
    f.sex, f.aliases = "female", {"MN9x": "MN9", "prefix:MN9y": "MN9"}
    f.sign = conn.sign.copy()
    f.sign[conn.select("nt:unclear")[:2]] = -1
    monkeypatch.setattr(cli, "load_connectome", lambda *a, **k: f)
    return f


def test_hints_stay_on_the_female_fly(capsys, female):
    with pytest.raises(SystemExit, match="No neurons match 'pIP11' in the female fly.*python fly_brain.py --female --find pIP11"):
        main(["--female", "--stim", "pIP11:60"])
    main(["--female", "--only", "silence", "--seeds", "1"])
    out = capsys.readouterr().out
    assert "Try the game's settings: python fly_brain.py --profile game --female" in out and "Then play: python fly_game.py --female" in out
    with pytest.raises(SystemExit, match="not available for the female fly"):     # refused before the name is looked up
        main(["--female", "--lines", "pIP11"])


def test_find_lists_the_aliases(capsys, female):
    main(["--female", "--find", "mn9"])
    out = capsys.readouterr().out
    assert "MN9x" in out and "(alias of MN9)" in out and "MN9y" not in out and "2 types match 'mn9'" in out


def test_genes_on_the_female_fly(capsys, female, conn, monkeypatch):
    main(["--female", "--genes"])
    out = capsys.readouterr().out
    n = conn.select("nt:unclear").size
    assert "Gene expression in the data (FlyWire annotation):" in out
    assert (f"{n:,} neurons have no confident transmitter prediction (labelled 'unclear'); each keeps its predicted "
            f"transmitter's sign ({n - 2:,} excitatory, 2 inhibitory).") in out
    monkeypatch.setattr(cli, "load_connectome", lambda *a, **k: conn)
    main(["--genes"])
    out = capsys.readouterr().out
    assert "(the MaleCNS annotation)" in out and f"{n:,} neurons have no confident transmitter prediction and count as excitatory" in out


def test_help_describes_every_option(capsys):
    with pytest.raises(SystemExit):
        main(["--help"])
    out = " ".join(capsys.readouterr().out.split())
    assert "--seed SEED first random seed" in out and out.count("implies --parts") == 5 and "(K = 1 to 2048)" in out


def test_python_dash_m_names_itself():
    import subprocess
    import sys
    from pathlib import Path
    r = subprocess.run([sys.executable, "-m", "virtual_fly", "--bogus"], capture_output=True, text=True,
                       cwd=Path(cli.__file__).resolve().parents[1])
    assert r.returncode == 2 and r.stderr.startswith("usage: python -m virtual_fly") and "python -m virtual_fly: error:" in r.stderr
