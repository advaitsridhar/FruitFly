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
from .wiring import LEVEL_ERROR, valid_level


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
    ap = argparse.ArgumentParser(description="Simulate a whole fruit-fly nervous system (MaleCNS v1.0; FlyWire 783 with --female).",
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
    ap.add_argument("--json", metavar="FILE",
                    help="write the results as JSON: the experiments, or those of --stim, --sweep, --lesion, --trace, --inputs/--outputs")
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
    ap.add_argument("--gain", type=float, default=None, help="global synaptic gain (default 0.65 for the male fly, 1.0 = the paper's value for the female fly)")
    ap.add_argument("--kenyon-gain", type=float, default=None, help="input gain of Kenyon cells (0.25 pure, 1.0 game)")
    ap.add_argument("--fatigue", type=float, default=None, metavar="MV", help="threshold increase per spike, fading over 2 s")
    ap.add_argument("--std", metavar="U:TAU_MS", help="short-term synaptic depression, e.g. 0.1:150")
    ap.add_argument("--noise", metavar="HZ:MV",
                    help="background kicks per neuron: HZ kicks a second, each MV into the synaptic input (a 1 mV kick "
                         "lifts the membrane by at most 0.16 mV; the threshold is 7 mV above rest). Parts list off: 2:10 "
                         "fires nothing, 5:15 about 600-900 spikes/s, 2:20 runs away; with the parts list on 2:1 already "
                         "runs away (docs/SCIENCE.md 3.4)")
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
                    help="parts list: ignore the receptors each target type expresses (one net sign per modulator; "
                         "implies --parts)")
    ap.add_argument("--one-sign-rule", action="store_true",
                    help="parts list: a target whose receptors are unknown feels each tone with the modulator's one net "
                         "sign, as in v2.7 (default since v2.8: it feels no tone; docs/SCIENCE.md 8.4; implies --parts)")
    ap.add_argument("--global-apl", action="store_true",
                    help="parts list: APL releases as one cell, the same everywhere, instead of following the Kenyon cells "
                         "active around each target (Amin et al. 2020; implies --parts)")
    ap.add_argument("--silence", metavar="SPEC", default="", help='block the output of a population, e.g. "class:ALLN" or "MN9"')
    ap.add_argument("--modulate", metavar="SPEC:FACTOR", default="", help='scale the output of a population, e.g. "LB3b,LB3c:1.5"')
    ap.add_argument("--record", metavar="FILE.npz", help="with --stim: save every spike (time_ms, neuron) to this file")
    ap.add_argument("--genes", action="store_true",
                    help="list the gene-expression populations in the data (fruitless, doublesex, transmitter genes) with FlyBase links")
    ap.add_argument("--lines", metavar="SPEC", help="driver lines whose expression images match these neurons (NeuronBridge; needs internet)")
    ap.add_argument("--driver", metavar="LINE", help="MaleCNS neurons a driver line labels, e.g. SS02385 (NeuronBridge; needs internet)")
    ap.add_argument("--grow", metavar="LEVEL",
                    help="run everything on a fly grown from its wiring rules: type, class or bottleneck:K (K = 1 to 2048)")
    ap.add_argument("--grow-seed", type=int, default=1, help="which individual to grow")
    ap.add_argument("--genome-sweep", metavar="LEVELS", nargs="?", const="real,type,class,bottleneck:64",
                    help='grow a fly at each level (comma-separated; default "real,type,class,bottleneck:64") and table which experiments survive')
    ap.add_argument("--female", action="store_true",
                    help="the female fly: FlyWire's whole-brain connectome (release 783), built on first use from its public "
                         "sources (needs pyarrow); no nerve cord, so experiments on leg and wing motor neurons are n/a")
    ap.add_argument("--top", type=int, default=15, help="how many rows to show in rankings")
    ap.add_argument("--seed", type=int, default=0,
                    help="first random seed (default 0): the experiments and --genome-sweep use SEED to SEED+SEEDS-1; "
                         "--stim, --sweep and --lesion use SEED")
    args = ap.parse_args(argv)
    _check_args(ap, args)

    conn = load_connectome(female=args.female)
    if args.find:
        hits = conn.find_types(args.find)
        types = set(conn.tables["types"])
        for t, c in hits[:200]:
            alias = conn.aliases.get(t, "") if t not in types else ""     # the kit's name for cells the file calls otherwise
            note = f"  (alias of {'a list of neuron ids' if alias.startswith('body:') else alias})" if alias else ""
            print(f"  {t:28} {c:6} neurons{note}")
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
        cmp = compare(load_connectome(quiet=True, female=args.female), conn) if rules is not None else {}
        print(f"grown a fly from its {args.grow} wiring rules (seed {args.grow_seed}) in {time.time() - t0:.0f} s: "
              f"{conn.n_edges:,} connections, {int(conn.n_syn.sum()):,} synapses"
              + (f", {100 * cmp.get('shared_connections_fraction', 0):.0f}% shared with the real wiring" if cmp else "")
              + (f"; rules: {rules.summary()}" if rules is not None else ""))
    if args.genes:
        from .genetics import summary
        g = summary(conn)
        print(f"\nGene expression in the data ({'FlyWire' if conn.sex == 'female' else 'the MaleCNS'} annotation):")
        for e in g["expression"]:
            high = f"{e['high']:,} high confidence" if e["high"] is not None else ""
            print(f"  {e['label']:26} {e['n']:7,} neurons in {e['types']:5,} types  {high:24} {e['spec']:22} {e['flybase'] or ''}")
        print("\nTransmitter genes (the transmitter is predicted from the synapses' appearance):")
        for t in g["transmitters"]:
            print(f"  {t['nt']:14} {t['n']:8,} neurons  {100 * t['synapse_share']:5.1f} % of synapses  {'excites ' if t['sign'] > 0 else 'inhibits'}  "
                  + ", ".join(f"{x['symbol']} {x['flybase']}" for x in t["genes"]))
        if g["unclear_inhibitory"]:              # FlyWire keeps the prediction's sign under the "unclear" label
            print(f"\n  {g['unclear']:,} neurons have no confident transmitter prediction (labelled 'unclear'); each keeps its "
                  f"predicted transmitter's sign ({g['unclear'] - g['unclear_inhibitory']:,} excitatory, "
                  f"{g['unclear_inhibitory']:,} inhibitory).\n  {g['source']}")
        else:
            print(f"\n  {g['unclear']:,} neurons have no confident transmitter prediction and count as excitatory.\n  {g['source']}")
        return
    if args.lines or args.driver:
        from .genetics import NeuronBridge, NeuronBridgeError, _malecns_only
        nb = NeuronBridge()
        try:
            _malecns_only(conn)                  # the female fly is refused before a name is looked up
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
        except (NeuronBridgeError, ValueError) as e:
            raise SystemExit(str(e))
        return
    if args.inputs or args.outputs:
        out = {}
        for spec, direction in ((args.inputs, "in"), (args.outputs, "out")):
            if not spec:
                continue
            check(conn, spec)
            rows = strongest_partners(conn, spec, direction, top=args.top)
            out[f"{direction}puts"] = {"spec": spec, "neurons": conn.count(spec), "rows": rows}
            title = "inputs of" if direction == "in" else "outputs of"
            # the share is of the partner's own traffic: how much of its output reaches the population (inputs),
            # or how much of its input comes from the population (outputs)
            share_of = "% of its output" if direction == "in" else "% of its input"
            print(f"\nStrongest {title} {spec} ({conn.count(spec)} neurons):")
            print(f"  {'type':28} {'synapses':>9} {'cells':>6} {'nt':17} {share_of:>15}")
            for r in rows:
                name = f"{r['type']}/{r['side']}" if r["side"] else r["type"]
                share = "" if r["fraction"] is None else f"{100 * r['fraction']:.1f}%"
                print(f"  {name:28} {r['synapses']:9} {r['neurons']:6} {r['nt'] + (' (-)' if r['sign'] < 0 else ' (+)'):17} {share:>15}")
        if args.json:
            with open(args.json, "w") as f:
                json.dump(out, f, indent=1)
            print(f"wrote {args.json}")
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
    if args.parts or args.part or args.curated or args.no_receptor_signs or args.global_apl or args.one_sign_rule:
        from .parts import PartsList, parse_param
        try:
            pl = PartsList(curated=args.curated or "modulators", receptor_signs=not args.no_receptor_signs,
                           unknown_sign=1.0 if args.one_sign_rule else 0.0)
            overrides["parts"] = (pl if not args.global_apl else replace(pl, local=())).with_params(args.part)
        except ValueError as e:
            raise SystemExit(f"--part: {e}")
        for s in args.part:                      # an override that matches no neuron would change nothing
            check(conn, parse_param(s).spec)
        c = overrides["parts"].compile(conn).counts
        cur = c["curated"]
        with_data = max((r["with_data"] for r in c["receptor_signs"]["coverage"]), default=0)
        by_fact = sum(f["targets"] for f in c["receptor_signs"].get("facts", []))    # signed by a fact, not the atlas
        print(f"parts list on: {c['modulatory_neurons']:,} modulatory neurons ({', '.join(m['nt'] for m in c['modulators'])}) act through "
              f"slow tones on {c['modulated_targets']:,} targets; {c['graded_neurons']:,} graded cells"
              + (f"; curated transmitters ({cur['policy']}): {cur['neurons']:,} neurons in {cur['types']:,} types changed" if cur.get("neurons") else "")
              + (f"; receptor signs on {with_data:,} modulated targets" if with_data else "")
              + ("; the one-sign rule for targets without receptor data (v2.7)" if args.one_sign_rule
                 else "" if args.no_receptor_signs
                 else f"; no tone on the {c['modulated_targets'] - with_data - by_fact:,} targets without receptor data")
              + "".join((f"; receptors from the literature for {f['spec']} ({f['what']})" if f["receptors"]
                         else f"; from the literature: {f['what']} ({f['label']})")
                        for f in c["receptor_signs"].get("facts", []) if f["neurons"])
              + "".join(f"; {x['spec']} releases locally ({x['compartments']} compartments, by "
                        f"{'neuPrint region' if x.get('mode') == 'regions' else 'lobe'})" for x in c["local"])
              + (f"; overrides: {', '.join(p['spec'] + ' (' + format(p['neurons'], ',') + ' neurons) -> ' + ', '.join(f'{k} {v}' for k, v in p.items() if k in ('theta_mv', 'graded') and v is not None) for p in c['params'])}" if c["params"] else ""))
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
        skipped = [e for e in E.CLASSIC + E.EXTENDED if e.profile is not None and e.profile != args.profile]
        if skipped:                              # survival() leaves them out silently; say so, as run_all() does
            print(f"\n(skipped {len(skipped)} experiments whose ranges were measured with the '{skipped[0].profile}' profile: "
                  f"{', '.join(e.name for e in skipped)}; run them with --profile {skipped[0].profile})")
        return
    brain = build_brain(conn, args.profile, **overrides)
    for spec in filter(None, (x.strip() for x in args.silence.split(";"))):
        print(f"silencing {spec}: {brain.silence(check_spec(conn, spec))} neurons")
    for item in filter(None, (x.strip() for x in args.modulate.split(";"))):
        spec, _, factor = item.rpartition(":")
        print(f"modulating {spec} x{float(factor):g}: {brain.modulate(check_spec(conn, spec), float(factor))} neurons")
    watch = [w.strip() for w in args.watch.replace(";", ",").split(",") if w.strip()]
    for spec in watch:                           # a misspelt name would read 0 Hz: refuse it before simulating
        check(conn, spec)

    if args.stim:
        for spec, hz in parse_stim(args.stim):
            head, _, rate = spec.rpartition(":")
            if not conn.count(spec) and conn.count(head):          # "MDN:abc": a population, then no rate
                _usage_error(ap, f"--stim: '{rate}' is not a rate in Hz; use SPEC:HZ, e.g. \"MDN:60\"")
            brain.stimulate(check_spec(conn, spec), hz)
            print(f"stimulating {spec} ({conn.count(spec)} neurons) at {hz:g} Hz")
        if args.record:
            brain.start_recording()
        t0 = time.time()
        brain.run(args.ms)
        print(f"simulated {args.ms:g} ms in {time.time() - t0:.1f} s; {brain.total_spikes:,} spikes in total")
        rates = {}
        for spec in watch:
            rates[spec] = brain.rate(spec)
            print(f"  {spec:28} {conn.count(spec):5} neurons  {rates[spec]:7.1f} Hz")
        top = [] if watch else brain.top_types(args.top, exclude_stimulated=True)
        if not watch:
            print("  most active cell types (mean Hz per neuron; stimulated ones excluded):")
            for row in top:
                name = f"{row['type']}/{row['side']}" if row["side"] else row["type"]
                print(f"    {row['hz']:6.0f} Hz  {name:28} {row['active']}/{row['neurons']} neurons active")
        if args.record:
            t_ms, idx = brain.recording_arrays(brain.stop_recording())
            np.savez_compressed(args.record, time_ms=t_ms, neuron=idx, body_id=conn.body_id[idx])
            print(f"saved {idx.size:,} spikes to {args.record}")
        if args.json:
            with open(args.json, "w") as f:
                json.dump({"stimulus": args.stim, "ms": args.ms, "rates": rates, "top_types": top, "settings": brain.settings()},
                          f, indent=1)
            print(f"wrote {args.json}")
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
        spec = next((r.spec for r in exp.readouts if r.spec == readout or r.label == readout), None)
        if spec is None:
            _usage_error(ap, f"--readout: '{readout}' is not a readout of '{exp.name}'; choose one of: "
                             + ", ".join(r.spec for r in exp.readouts))
        check(conn, spec)
        candidates = [c.strip() for c in args.candidates.split(",") if c.strip()]
        for c in candidates:
            check(conn, c)
        if not candidates:
            # one trace per stimulus population: traced from their union, the strongest partial routes of some
            # (the pharyngeal and leg sugar cells) fill the search beam and crowd out every route of another to the readout
            paths = [p for s in exp.stimulus if conn.count(s) for p in trace(conn, s, spec, max_hops=args.hops, top=20)]
            stimulated = conn.select(",".join(exp.stimulus))
            candidates = [name for name, _ in relay_ranking(paths)
                          if not np.isin(conn.select(name), stimulated).all()][:10]
            if not candidates:
                raise SystemExit(f"no wiring route from the stimulus of '{exp.name}' to {readout} within {args.hops} hops; "
                                 f"name the populations to silence with --candidates")
            print(f"lesion candidates from the wiring between the stimulus and {readout}: {', '.join(candidates)}")
        rows = E.lesion_scan(brain, exp, candidates, readout, seed=args.seed)
        print(f"\n{exp.name}: {readout} with each population silenced (baseline {rows[0]['baseline']:.1f} Hz)")
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
    done = [r for r in results if r.ok is not None]
    bad = [r for r in done if not r.ok]
    n_read = sum(sum(x.ok is not None for x in r.readouts) for r in done)
    n_ok = sum(sum(bool(x.ok) for x in r.readouts) for r in done)
    fragile = [r for r in results if r.fragile]
    na = [r for r in results if r.ok is None]
    stim = [r for r in done if r.after_per_seed]
    loose = [r for r in stim if r.after_sps >= 50000]
    print(f"\n{n_ok}/{n_read} readouts in the expected range ({len(done) - len(bad)}/{len(done)} experiments"
          + (f"; {len(fragile)} pass on the mean but miss on some seed: {', '.join(r.name for r in fragile)}" if fragile else "")
          + (f"; {len(na)} cannot be done on this fly: {', '.join(r.name for r in na)}" if na else "")
          + ")."
          + (f" After the stimulus, {len(loose)} of {len(stim)} leave a runaway loop on at least one seed." if loose else ""))
    if any(E.in_margin(x) for r in done for x in r.readouts):
        print("(+N) after a range: a rate passes up to max(1 Hz, 15 %) above the top of its range, on the mean and on each "
              "seed (docs/SCIENCE.md section 2).")
    if args.json:
        E.save_json(results, args.json, brain)
        print(f"wrote {args.json}")
    same_fly = ((" --female" if args.female else "") + (f" --grow {args.grow}" if args.grow else "")
                + (f" --grow-seed {args.grow_seed}" if args.grow and args.grow_seed != 1 else ""))
    if args.profile == "pure":
        print("Try the game's settings: python fly_brain.py --profile game" + same_fly)
    print("Then play: python fly_game.py" + same_fly)


def _usage_error(ap, message):
    """Stop with one line and exit code 2, as argparse does for a malformed option (without its 20-line usage)."""
    ap.exit(2, f"{ap.prog}: error: {message}\n")


def _check_args(ap, args):
    """Refuse malformed option values before anything is loaded or simulated."""
    def numbers(text, n):                        # "0.1:150" -> [0.1, 150.0]; None unless it is n numbers
        try:
            vals = [float(x) for x in text.split(":")]
        except ValueError:
            return None
        return vals if len(vals) == n else None
    if args.std and numbers(args.std, 2) is None:
        _usage_error(ap, f"--std wants U:TAU_MS, e.g. 0.1:150 (not '{args.std}')")
    if args.noise and numbers(args.noise, 2) is None:
        _usage_error(ap, f"--noise wants HZ:MV, e.g. 5:15 (not '{args.noise}')")
    for item in filter(None, (x.strip() for x in args.modulate.split(";"))):
        spec, _, factor = item.rpartition(":")
        if not spec or numbers(factor, 1) is None:
            _usage_error(ap, f"--modulate wants SPEC:FACTOR, e.g. \"LB3b,LB3c:1.5\" (not '{item}')")
    if args.sweep:
        spec, *rest = args.sweep.rsplit(":", 3)
        if len(rest) != 3 or not spec or numbers(":".join(rest[:2]), 2) is None or not rest[2].strip().isdigit() \
                or int(rest[2]) < 1:
            _usage_error(ap, f"--sweep wants SPEC:LO:HI:N (N rates from LO to HI Hz), e.g. \"LB3b,LB3c:0:200:9\" (not '{args.sweep}')")
    levels = [x.strip() for x in (args.genome_sweep or "").split(",") if x.strip()]
    if args.genome_sweep is not None and not levels:
        _usage_error(ap, "--genome-sweep wants comma-separated levels, e.g. real,type,class,bottleneck:64")
    grow = [("--grow", args.grow)] if args.grow else []
    for option, level in grow + [("--genome-sweep", lv) for lv in levels]:
        if not valid_level(level):               # K is bounded: see wiring.MAX_RANK
            _usage_error(ap, f"{option}: '{level}': {LEVEL_ERROR}")
    if args.seeds < 1:
        _usage_error(ap, f"--seeds must be at least 1 (not {args.seeds})")
    if args.record and not args.stim:
        _usage_error(ap, "--record FILE.npz saves the spikes of a --stim run; add --stim")
    if args.only:                                # the same match as experiments.run_all()
        exps = E.all_experiments()
        if not any(args.only.lower() in e.name.lower() or args.only.lower() in " ".join(e.tags) for e in exps):
            _usage_error(ap, f"no experiment's name or tag contains '{args.only}'; the tags are "
                             + ", ".join(sorted({t for e in exps for t in e.tags})) + "; the names are "
                             + "; ".join(e.name for e in exps))


def check(conn, spec):
    """The neurons ``spec`` selects; or stop with one line, and a hint to search the names in the same fly."""
    female = getattr(conn, "sex", "male") == "female"
    try:
        idx = conn.select(spec)
        if idx.size:
            return idx
        problem = f"No neurons match '{spec}'" + (" in the female fly (no nerve cord, no male-specific cells)" if female else "") + "."
    except ValueError as e:
        problem = str(e).rstrip(".") + "."
    if spec.split(":")[0] in ("body", "index", "hex", "regex"):
        raise SystemExit(problem)                # a number or a pattern, not a name: a name search would not help
    word = spec.split(":")[-1].split("/")[0].split(",")[0]
    raise SystemExit(f"{problem} Search for names with: python fly_brain.py{' --female' if female else ''} --find {word}")


def check_spec(conn, spec):
    check(conn, spec)
    return spec


if __name__ == "__main__":
    main()
