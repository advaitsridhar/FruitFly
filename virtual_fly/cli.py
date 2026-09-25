"""
Command line for the brain simulator (``python fly_brain.py ...`` or ``python -m virtual_fly ...``).

    python fly_brain.py                                   # run every validated experiment
    python fly_brain.py --only taste --seeds 1            # a subset, one seed (quick; the default is 5)
    python fly_brain.py --find DNa                        # search cell types by name
    python fly_brain.py --stim "MDN:60" --watch "MDN,DNp09"
    python fly_brain.py --stim "LC4/R,LPLC2/R:150"        # no --watch: shows the most active types
    python fly_brain.py --trace LC10a/L DNa02/L           # strongest wiring routes between two populations
    python fly_brain.py --inputs MN9 --outputs GNG232     # strongest partners of a population
    python fly_brain.py --sweep "LB3b,LB3c:0:200:9" --watch MN9      # dose-response curve
    python fly_brain.py --lesion "Sugar" --readout MN9 --candidates GNG232,DNg67,DNge080
    python fly_brain.py --profile pure --json results.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace

import numpy as np

from . import experiments as E
from .connectome import load_connectome
from .pathways import relay_ranking, strongest_partners, trace
from .settings import PROFILES, build_brain


def parse_stim(text: str, default_hz: float = 80.0) -> list[tuple[str, float]]:
    out = []
    for item in filter(None, (x.strip() for x in text.split(";"))):
        spec, sep, hz = item.rpartition(":")
        try:
            hz = float(hz)
        except ValueError:                       # no rate given, e.g. --stim MDN
            spec, hz = item, default_hz
        if not sep:
            spec = item
        out.append((spec, hz))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="Simulate the whole male fruit fly nervous system (MaleCNS v1.0).",
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--find", metavar="TEXT", help="list neuron types whose name contains TEXT")
    ap.add_argument("--info", metavar="SPEC", help="describe the neurons matching SPEC (first 30)")
    ap.add_argument("--stim", metavar="SPEC:HZ[;SPEC:HZ]", help='stimulate populations, e.g. "MDN:60" or "LC4/R,LPLC2/R:150"')
    ap.add_argument("--watch", metavar="SPEC[;SPEC]", default="", help='populations to report, separated by ";" or ","')
    ap.add_argument("--ms", type=float, default=500, help="how long to simulate with --stim (default 500 ms)")
    ap.add_argument("--only", metavar="TEXT", help="run only experiments whose name or tag contains TEXT")
    ap.add_argument("--seeds", type=int, default=5,
                    help="repeat each experiment with this many random seeds (default 5: the verdict is the mean, and a "
                         "readout that passes on the mean while some seed on its own misses is reported)")
    ap.add_argument("--json", metavar="FILE", help="write experiment results (or --stim rates) as JSON")
    ap.add_argument("--trace", nargs=2, metavar=("FROM", "TO"), help="strongest wiring routes between two populations")
    ap.add_argument("--hops", type=int, default=4, help="maximum path length for --trace (default 4)")
    ap.add_argument("--avoid", metavar="SPEC", help="route --trace around these types (a virtual lesion)")
    ap.add_argument("--inputs", metavar="SPEC", help="strongest presynaptic types of a population")
    ap.add_argument("--outputs", metavar="SPEC", help="strongest postsynaptic types of a population")
    ap.add_argument("--sweep", metavar="SPEC:LO:HI:N", help="dose-response: stimulate SPEC at N rates from LO to HI Hz")
    ap.add_argument("--lesion", metavar="EXPERIMENT", help="silence each --candidates population during this experiment")
    ap.add_argument("--readout", metavar="SPEC", help="the readout to track for --lesion")
    ap.add_argument("--candidates", metavar="SPEC[,SPEC]", default="",
                    help="populations to lesion (default: relays found by --trace between the stimulus and the readout)")
    ap.add_argument("--profile", choices=sorted(PROFILES), default="pure",
                    help="model profile: pure (the paper, default here), game (what the game runs), brakes")
    ap.add_argument("--dt", type=float, default=0.5, help="time step in ms (0.1 = the paper's Brian2 default, slower)")
    ap.add_argument("--backend", choices=("auto", "numpy", "numba"), default="auto",
                    help="integrator: compiled numba kernels when numba is installed (auto), or plain NumPy; same spikes either way")
    ap.add_argument("--gain", type=float, default=None, help="global synaptic gain (default 0.65)")
    ap.add_argument("--kenyon-gain", type=float, default=None, help="input gain of Kenyon cells (0.25 pure, 1.0 game)")
    ap.add_argument("--fatigue", type=float, default=None, metavar="MV", help="threshold increase per spike, fading over 2 s")
    ap.add_argument("--std", metavar="U:TAU_MS", help="short-term synaptic depression, e.g. 0.1:150")
    ap.add_argument("--noise", metavar="HZ:MV", help="background kicks per neuron, e.g. 2:1.0")
    ap.add_argument("--jitter", type=float, default=None, metavar="MV", help="per-neuron threshold jitter (sd, mV)")
    ap.add_argument("--parts", action="store_true",
                    help="the genes as each neuron's parts list: dopamine, octopamine and serotonin act through slow tones, "
                         "the optic lobe's graded cell types transmit below threshold (parts.py)")
    ap.add_argument("--part", metavar="SPEC:theta=MV[,graded=0/1]", action="append", default=[],
                    help='a per-type override in the parts list, e.g. "class:Kenyon_Cell:theta=10" (implies --parts; repeatable)')
    ap.add_argument("--curated", choices=("off", "modulators", "all"), default=None,
                    help="how far the parts list follows Virtual Fly Brain's curated transmitters over the MaleCNS prediction "
                         "(default modulators: fill 'unclear' predictions and correct which neurons are modulators; "
                         "all: also flip the sign of a confident fast prediction; implies --parts)")
    ap.add_argument("--no-receptor-signs", action="store_true",
                    help="parts list: ignore the receptors each target type expresses (one net sign per modulator)")
    ap.add_argument("--global-apl", action="store_true",
                    help="parts list: APL releases as one cell, the same everywhere, instead of following the Kenyon cells "
                         "active around each target (Amin et al. 2020)")
    ap.add_argument("--silence", metavar="SPEC", default="", help='block the output of a population, e.g. "class:ALLN" or "MN9"')
    ap.add_argument("--modulate", metavar="SPEC:FACTOR", default="", help='scale the output of a population, e.g. "LB3b,LB3c:1.5"')
    ap.add_argument("--record", metavar="FILE.npz", help="with --stim: save every spike (time_ms, neuron) to this file")
    ap.add_argument("--genes", action="store_true",
                    help="list the gene-expression populations in the data (fruitless, doublesex, transmitter genes) with FlyBase links")
    ap.add_argument("--lines", metavar="SPEC", help="driver lines whose expression images match these neurons (NeuronBridge; needs internet)")
    ap.add_argument("--driver", metavar="LINE", help="MaleCNS neurons a driver line labels, e.g. SS02385 (NeuronBridge; needs internet)")
    ap.add_argument("--grow", metavar="LEVEL", help="run everything on a fly grown from its wiring rules: type, class or bottleneck:K")
    ap.add_argument("--grow-seed", type=int, default=1, help="which individual to grow")
    ap.add_argument("--genome-sweep", metavar="LEVELS", nargs="?", const="real,type,class,bottleneck:64",
                    help='grow a fly at each level (comma-separated; default "real,type,class,bottleneck:64") and table which experiments survive')
    ap.add_argument("--top", type=int, default=15, help="how many rows to show in rankings")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    conn = load_connectome()
    if args.find:
        hits = conn.find_types(args.find)
        for t, c in hits[:200]:
            print(f"  {t:28} {c:6} neurons")
        print(f"{len(hits)} types match '{args.find}'")
        return
    if args.info:
        idx = check(conn, args.info)
        for i in idx[:30]:
            print(" ", conn.describe(i))
        if idx.size > 30:
            print(f"  ... {idx.size} neurons in total")
        return
    if args.grow:
        from .wiring import compare, grow_level
        t0 = time.time()
        conn, rules = grow_level(conn, args.grow, args.grow_seed)
        cmp = compare(load_connectome(quiet=True), conn) if rules is not None else {}
        print(f"grown a fly from its {args.grow} wiring rules (seed {args.grow_seed}) in {time.time() - t0:.0f} s: "
              f"{conn.n_edges:,} connections, {int(conn.n_syn.sum()):,} synapses"
              + (f", {100 * cmp.get('shared_connections_fraction', 0):.0f}% shared with the real wiring" if cmp else "")
              + (f"; rules: {rules.summary()}" if rules is not None else ""))
    if args.genes:
        from .genetics import summary
        g = summary(conn)
        print("\nGene expression in the data (the MaleCNS annotation):")
        for e in g["expression"]:
            high = f"{e['high']:,} high confidence" if e["high"] is not None else ""
            print(f"  {e['label']:26} {e['n']:7,} neurons in {e['types']:5,} types  {high:24} {e['spec']:22} {e['flybase'] or ''}")
        print("\nTransmitter genes (the transmitter is predicted from the synapses' appearance):")
        for t in g["transmitters"]:
            print(f"  {t['nt']:14} {t['n']:8,} neurons  {100 * t['synapse_share']:5.1f} % of synapses  {'excites ' if t['sign'] > 0 else 'inhibits'}  "
                  + ", ".join(f"{x['symbol']} {x['flybase']}" for x in t["genes"]))
        print(f"\n  {g['unclear']:,} neurons have no confident transmitter prediction and count as excitatory.\n  {g['source']}")
        return
    if args.lines or args.driver:
        from .genetics import NeuronBridge, NeuronBridgeError
        nb = NeuronBridge()
        try:
            if args.lines:
                r = nb.lines_for(conn, check_spec(conn, args.lines))
                print(f"\nDriver lines matching {r['spec']} ({r['n']:,} neurons; {len(r['sampled'])} searched: bodies "
                      f"{', '.join(map(str, r['sampled']))}; NeuronBridge data {r['version']}):")
                for l in r["lines"]:
                    more = f"matches {l['neurons']} of them" if l["neurons"] > 1 else ""
                    print(f"  {l['line']:12} score {l['score']:9,.0f}  {more:22} {l['library']}")
                if r["unmatched"]:
                    print(f"  (not in NeuronBridge: bodies {', '.join(map(str, r['unmatched']))})")
            if args.driver:
                r = nb.neurons_for_line(conn, args.driver)
                print(f"\nMaleCNS neurons matching line {r['line']} ({r['library']}; {r['searched']} of {r['images']} images searched):")
                for n in r["neurons"][:args.top]:
                    kit = (n["type"] + ("/" + n["side"] if n["side"] else "")) if n["in_kit"] else "(not in this data)"
                    print(f"  body {n['body']:>9}  score {n['score']:9,.0f}  {kit:22} NeuronBridge type {n['nb_type'] or '?'}")
                if r["spec"]:
                    print(f"  spec for --stim / --silence: {r['spec']}")
        except NeuronBridgeError as e:
            raise SystemExit(str(e))
        return
    if args.inputs or args.outputs:
        for spec, direction in ((args.inputs, "in"), (args.outputs, "out")):
            if not spec:
                continue
            check(conn, spec)
            rows = strongest_partners(conn, spec, direction, top=args.top)
            title = "inputs of" if direction == "in" else "outputs of"
            print(f"\nStrongest {title} {spec} ({conn.count(spec)} neurons):")
            print(f"  {'type':28} {'synapses':>9} {'cells':>6} {'nt':14} {'share':>7}")
            for r in rows:
                name = f"{r['type']}/{r['side']}" if r["side"] else r["type"]
                share = "" if r["fraction"] is None else f"{100 * r['fraction']:.1f}%"
                print(f"  {name:28} {r['synapses']:9} {r['neurons']:6} {r['nt'] + (' (-)' if r['sign'] < 0 else ' (+)'):14} {share:>7}")
        return
    if args.trace:
        src, dst = args.trace
        check(conn, src), check(conn, dst)
        t0 = time.time()
        paths = trace(conn, src, dst, max_hops=args.hops, top=args.top, avoid=args.avoid)
        print(f"\n{len(paths)} strongest routes from {src} to {dst} (up to {args.hops} hops, {time.time() - t0:.1f} s):")
        for p in paths:
            print("  " + p.describe())
        relays = relay_ranking(paths)
        if relays:
            print("\nRelays carrying the most of these routes (lesion candidates):")
            for name, score in relays[:10]:
                print(f"  {name:28} {score:.4f}")
        if args.json:
            with open(args.json, "w") as f:
                json.dump([p.to_dict() for p in paths], f, indent=1)
        return

    overrides = {"dt": args.dt, "seed": args.seed, "backend": args.backend}
    for key, val in (("gain", args.gain), ("kenyon_gain", args.kenyon_gain), ("fatigue_mv", args.fatigue),
                     ("threshold_jitter", args.jitter)):
        if val is not None:
            overrides[key] = val
    if args.std:
        u, tau = (float(x) for x in args.std.split(":"))
        overrides.update(std_u=u, std_tau_ms=tau)
    if args.noise:
        hz, mv = (float(x) for x in args.noise.split(":"))
        overrides.update(noise_hz=hz, noise_mv=mv)
    if args.parts or args.part or args.curated or args.no_receptor_signs or args.global_apl:
        from .parts import PartsList
        try:
            pl = PartsList(curated=args.curated or "modulators", receptor_signs=not args.no_receptor_signs)
            overrides["parts"] = (pl if not args.global_apl else replace(pl, local=())).with_params(args.part)
        except ValueError as e:
            raise SystemExit(f"--part: {e}")
        c = overrides["parts"].compile(conn).counts
        cur = c["curated"]
        with_data = max((r["with_data"] for r in c["receptor_signs"]["coverage"]), default=0)
        print(f"parts list on: {c['modulatory_neurons']:,} modulatory neurons ({', '.join(m['nt'] for m in c['modulators'])}) act through "
              f"slow tones on {c['modulated_targets']:,} targets; {c['graded_neurons']:,} graded cells"
              + (f"; curated transmitters ({cur['policy']}): {cur['neurons']:,} neurons in {cur['types']:,} types changed" if cur.get("neurons") else "")
              + (f"; receptor signs on {with_data:,} modulated targets" if with_data else "")
              + "".join(f"; receptors from the literature for {f['spec']} ({', '.join(f['receptors'])})" for f in c["receptor_signs"].get("facts", []) if f["neurons"])
              + "".join(f"; {x['spec']} releases locally ({len(x['groups'])} compartments)" for x in c["local"])
              + (f"; overrides: {', '.join(p['spec'] + ' -> ' + ', '.join(f'{k} {v}' for k, v in p.items() if k in ('theta_mv', 'graded') and v is not None) for p in c['params'])}" if c["params"] else ""))
    if args.genome_sweep:
        from .experiments import survival
        from .wiring import compare, grow_level
        levels = [x.strip() for x in args.genome_sweep.split(",") if x.strip()]
        cache: dict = {}
        table: dict[str, list[dict]] = {}
        for level in levels:
            t0 = time.time()
            c2, rules = grow_level(conn, level, args.grow_seed, rules_cache=cache)
            cmp = compare(conn, c2) if rules is not None else {}
            rows = survival(build_brain(c2, args.profile, **overrides), profile=args.profile,
                            seeds=tuple(range(args.seed, args.seed + args.seeds)))
            table[level] = rows
            ok = sum(1 for r in rows if r["ok"]); tested = sum(1 for r in rows if r["ok"] is not None)
            print(f"{level:16} {ok:2d} / {tested} experiments survive   ({c2.n_edges:,} connections"
                  + (f", {100 * cmp.get('shared_connections_fraction', 0):.0f}% shared with the real wiring" if cmp else "")
                  + f"; {time.time() - t0:.0f} s)")
        names = [r["name"] for r in next(iter(table.values()))]
        print(f"\n{'experiment':36}" + "".join(f"{lv[:14]:>16}" for lv in table))
        for name in names:
            cells = []
            for lv in table:
                r = next((x for x in table[lv] if x["name"] == name), None)
                cells.append("n/a" if r is None or r["ok"] is None else "ok" if r["ok"] else f"{sum(x['ok'] for x in r['readouts'])}/{len(r['readouts'])}")
            print(f"{name[:36]:36}" + "".join(f"{c:>16}" for c in cells))
        return
    brain = build_brain(conn, args.profile, **overrides)
    for spec in filter(None, (x.strip() for x in args.silence.split(";"))):
        print(f"silencing {spec}: {brain.silence(check_spec(conn, spec))} neurons")
    for item in filter(None, (x.strip() for x in args.modulate.split(";"))):
        spec, _, factor = item.rpartition(":")
        print(f"modulating {spec} x{float(factor):g}: {brain.modulate(check_spec(conn, spec), float(factor))} neurons")
    watch = [w.strip() for w in args.watch.replace(";", ",").split(",") if w.strip()]

    if args.stim:
        for spec, hz in parse_stim(args.stim):
            brain.stimulate(check_spec(conn, spec), hz)
            print(f"stimulating {spec} ({conn.count(spec)} neurons) at {hz:g} Hz")
        if args.record:
            brain.start_recording()
        t0 = time.time()
        brain.run(args.ms)
        print(f"simulated {args.ms:g} ms in {time.time() - t0:.1f} s; {brain.total_spikes:,} spikes in total")
        rates = {}
        for spec in watch:
            rates[spec] = brain.rate(check_spec(conn, spec))
            print(f"  {spec:28} {conn.count(spec):5} neurons  {rates[spec]:7.1f} Hz")
        if not watch:
            print("  most active cell types (mean Hz per neuron; stimulated ones excluded):")
            for row in brain.top_types(args.top, exclude_stimulated=True):
                name = f"{row['type']}/{row['side']}" if row["side"] else row["type"]
                print(f"    {row['hz']:6.0f} Hz  {name:28} {row['active']}/{row['neurons']} neurons active")
        if args.record:
            t_ms, idx = brain.recording_arrays(brain.stop_recording())
            np.savez_compressed(args.record, time_ms=t_ms, neuron=idx, body_id=conn.body_id[idx])
            print(f"saved {idx.size:,} spikes to {args.record}")
        if args.json:
            with open(args.json, "w") as f:
                json.dump({"stimulus": args.stim, "ms": args.ms, "rates": rates, "settings": brain.settings()}, f, indent=1)
        return

    if args.sweep:
        spec, lo, hi, n = args.sweep.rsplit(":", 3)
        check_spec(conn, spec)
        rates = np.linspace(float(lo), float(hi), int(n))
        if not watch:
            raise SystemExit("--sweep needs --watch to say which populations to report")
        rows = E.sweep(brain, spec, rates, watch, ms=args.ms)
        print(f"\n  {'Hz in':>7}  " + "  ".join(f"{w:>12}" for w in watch))
        for r in rows:
            print(f"  {r['hz']:7.1f}  " + "  ".join(f"{r[w]:12.1f}" for w in watch))
        if args.json:
            with open(args.json, "w") as f:
                json.dump(rows, f, indent=1)
        return

    if args.lesion:
        exps = [e for e in E.all_experiments() if args.lesion.lower() in e.name.lower()]
        if not exps:
            raise SystemExit(f"no experiment matches '{args.lesion}'")
        exp = exps[0]
        readout = args.readout or exp.readouts[-1].spec
        candidates = [c.strip() for c in args.candidates.split(",") if c.strip()]
        if not candidates:
            src = ",".join(exp.stimulus)
            paths = trace(conn, src, readout, max_hops=args.hops, top=20)
            candidates = [name for name, _ in relay_ranking(paths)[:10]]
            print(f"lesion candidates from the wiring between the stimulus and {readout}: {', '.join(candidates)}")
        rows = E.lesion_scan(brain, exp, candidates, readout, seed=args.seed)
        print(f"\n{exp.name}: {readout} with each population silenced")
        print(f"  {'silenced':28} {'neurons':>7} {'Hz':>8} {'change':>8}")
        for r in rows:
            print(f"  {r['silenced']:28} {r['neurons']:7} {r['hz']:8.1f} {100 * r['change']:+7.0f}%")
        if args.json:
            with open(args.json, "w") as f:
                json.dump(rows, f, indent=1)
        return

    print(f"Running the validated experiments with the '{args.profile}' profile "
          f"(every neuron simulated, nothing trained)...")
    results = E.run_all(brain, only=args.only, seeds=tuple(range(args.seed, args.seed + args.seeds)),
                        profile=args.profile)
    bad = [r for r in results if not r.ok]
    n_read = sum(len(r.readouts) for r in results)
    n_ok = sum(sum(x.ok for x in r.readouts) for r in results)
    fragile = [r for r in results if r.fragile]
    print(f"\n{n_ok}/{n_read} readouts in the expected range ({len(results) - len(bad)}/{len(results)} experiments"
          + (f"; {len(fragile)} pass on the mean but miss on some seed: {', '.join(r.name for r in fragile)}" if fragile else "")
          + ").")
    if args.json:
        E.save_json(results, args.json, brain)
        print(f"wrote {args.json}")
    if args.profile == "pure":
        print("Try the game's settings: python fly_brain.py --profile game")
    print("Then play: python fly_game.py")


def check(conn, spec):
    try:
        idx = conn.select(spec)
        if idx.size:
            return idx
        problem = f"No neurons match '{spec}'."
    except ValueError as e:
        problem = str(e).rstrip(".") + "."
    word = spec.split(":")[-1].split("/")[0].split(",")[0]
    raise SystemExit(f"{problem} Search for names with: python fly_brain.py --find {word}")


def check_spec(conn, spec):
    check(conn, spec)
    return spec


if __name__ == "__main__":
    main()
