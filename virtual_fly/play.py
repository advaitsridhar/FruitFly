"""
Start the game: ``python fly_game.py`` or ``python -m virtual_fly.play``.

    --port 9000          use another port
    --host 0.0.0.0       let other computers on your network open the game (anyone on it can drive the fly)
    --no-browser         don't open a browser tab
    --profile pure       the paper's model exactly (expect runaway loops after bitter, dust and smells)
    --no-autopilot       start with the hand-built walking urge switched off
    --no-learning        start with mushroom-body plasticity switched off
    --no-columnar        don't drive the connectome's own T4/T5 motion-detector columns from the retina
    --noise 5:15         background kicks per neuron per second : kick size (2:1 fires nothing; with --parts even
                         2:1 runs away: docs/SCIENCE.md 3.4)
    --fast               brain time step 1 ms instead of 0.5 ms (about twice as fast; all six
                         classic experiments still pass)
    --parts              start with the genes as each neuron's parts list (the Genome card toggles it)
    --body physics       walk with NeuroMechFly v2 legs in MuJoCo instead of the drawn body (needs flygym;
                         runs at about a tenth of real time)
    --partner female     a second simulated fly in the dish, with a brain of its own (FlyWire's female, or
                         `male` for a second MaleCNS brain); each brain then runs in its own process
    --social LIST        which of the hand-built channels between the two flies are on (default
                         seen,song,contact,collide; also cva, mating:virgin, mating:mated, touch, cues)
"""

from __future__ import annotations

import argparse
import sys

from .connectome import load_connectome
from .game import Game
from .senses.social import CHANNELS, DEFAULT_CHANNELS, SocialConfig
from .server import serve
from .settings import PROFILES, build_brain


def port_number(text: str) -> int:
    port = int(text)
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError(f"{port} is not a port: use 1 to 65535")
    return port


def main(argv=None):
    try:
        _main(argv)
    except KeyboardInterrupt:                    # Ctrl+C while the fly is built (the game itself says "Bye!")
        raise SystemExit("\nStopped before the game started.")


def _main(argv=None):
    ap = argparse.ArgumentParser(description="Play with a fly driven by a whole connectome (MaleCNS v1.0; FlyWire 783 with --female).",
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--port", type=port_number, default=8765, help="the first port to try (1-65535; the next 19 are tried too)")
    ap.add_argument("--host", default="127.0.0.1",
                    help="address to listen on: 127.0.0.1 (default) is this computer only; 0.0.0.0 lets other computers "
                         "on your network open the game and drive the fly (there is no password)")
    ap.add_argument("--no-browser", action="store_true", help="don't open a browser tab automatically")
    ap.add_argument("--profile", choices=sorted(PROFILES), default="game")
    ap.add_argument("--pure", action="store_true", help="same as --profile pure")
    ap.add_argument("--fatigue", type=float, default=None, help="neuron fatigue in mV per spike (0 = off)")
    ap.add_argument("--noise", metavar="HZ:MV", default=None,
                    help="background kicks per neuron, HZ:MV (off by default): with the parts list off 2:1 fires nothing and "
                         "5:15 gives about 600-900 spikes/s; with --parts even 2:1 runs away (docs/SCIENCE.md 3.4)")
    ap.add_argument("--kenyon-gain", type=float, default=None)
    ap.add_argument("--no-autopilot", action="store_true", help="start with the hand-built walking urge switched off")
    ap.add_argument("--no-learning", action="store_true", help="switch mushroom-body plasticity off")
    ap.add_argument("--no-columnar", action="store_true",
                    help="don't drive the connectome's T4/T5 motion-detector columns from the retina")
    ap.add_argument("--dt", type=float, default=0.5, help="brain time step in ms (0.5 default; 1.0 = twice as fast, slightly coarser)")
    ap.add_argument("--fast", action="store_true", help="same as --dt 1.0: for computers that run the brain below real time")
    ap.add_argument("--backend", choices=("auto", "numpy", "numba", "cupy"), default="auto",
                    help="brain integrator: the compiled numba kernels when numba is installed (auto), plain NumPy, or the GPU (cupy: needs the cupy package and an NVIDIA GPU); same spikes every way")
    ap.add_argument("--grow", metavar="LEVEL", default=None,
                    help="start with a fly grown from its wiring rules: type, class or bottleneck:K, K = 1 to 2048 (the Genome card does the same)")
    ap.add_argument("--grow-seed", type=int, default=1, help="which individual to grow (any whole number)")
    ap.add_argument("--parts", action="store_true",
                    help="start with the parts list on: modulators as slow tones, graded optic-lobe cells (the Genome card toggles it)")
    ap.add_argument("--curated", choices=("off", "modulators", "all"), default=None,
                    help="the parts list's policy for Virtual Fly Brain's curated transmitters, used whenever the parts list "
                         "is on (now with --parts, or when switched on from the Genome card): off; modulators (default) = fill "
                         "'unclear' predictions and correct which neurons are modulators; all = the literature also wins over "
                         "confident fast predictions")
    ap.add_argument("--female", action="store_true",
                    help="play with the female fly: FlyWire's whole-brain connectome (release 783), built on first use "
                         "(needs pyarrow); no nerve cord and no computed column-by-column motion vision")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--body", choices=("drawn", "physics"), default="drawn",
                    help="drawn: the kinematic body (default); physics: NeuroMechFly v2 legs in MuJoCo (optional, needs flygym)")
    ap.add_argument("--stride-average", action="store_true",
                    help="physics body: the senses see the body's pose averaged over one stride (hand-built, a stand-in for "
                         "gaze stabilisation) instead of the stride-by-stride wobble")
    ap.add_argument("--brain-procs", choices=("auto", "on", "off", "server"), default="auto",
                    help="where the brain runs: auto (in this process for one fly; one process per brain with a partner; one "
                         "process for every brain with --backend cupy), on (its own process even for one fly), off (always in "
                         "this process), server (one brain process for every fly)")
    ap.add_argument("--partner", choices=("none", "female", "male"), default="none",
                    help="a second simulated fly in the dish, sensing the first only through the world: female = FlyWire's "
                         "whole brain (release 783, built on first use, needs pyarrow); male = a second MaleCNS brain; "
                         "with --female the protagonist is the female and --partner male gives her a male partner")
    ap.add_argument("--social", metavar="LIST", default=None,
                    help=f"with --partner: the channels between the flies that are on, a comma list (default {DEFAULT_CHANNELS}; "
                         f"the channels are {', '.join(CHANNELS)}, mating as mating:virgin or mating:mated)")
    ap.add_argument("--physics-levers", metavar="LIST", default=None,
                    help="physics body: speed levers to switch on, a comma list of " + ", ".join(
                        f"{k} ({v})" for k, v in __import__("virtual_fly.physics", fromlist=["LEVERS"]).LEVERS.items())
                    + "; each changes the physics and is measured before it is adopted (docs/SCIENCE.md 13.5); the default is "
                      "dedupe, which changes no number, and 'none' switches every lever off")
    ap.add_argument("--partner-body", choices=("drawn", "physics"), default=None,
                    help="with --partner: the partner's body; it is the protagonist's (--body): physics puts both flies in one "
                         "MuJoCo world, so --partner-body physics needs --body physics, and --partner-body drawn the drawn body")
    args = ap.parse_args(argv)
    # the partner's flags are checked before anything is loaded (and before the physics body's own check, which
    # stops with an install hint wherever flygym is missing)
    partner = None if args.partner == "none" else args.partner
    if args.social is not None and partner is None:
        ap.error("--social only applies with --partner female or male; add it, or leave out --social")
    if args.partner_body is not None and partner is None:
        ap.error("--partner-body only applies with --partner female or male; add it, or leave out --partner-body")
    if partner is not None and args.partner_body is not None and args.partner_body != args.body:
        ap.error(f"--partner-body {args.partner_body} with --body {args.body}: both flies have the same kind of body (with "
                 "--body physics both are NeuroMechFly bodies in one MuJoCo world); leave out --partner-body, or make them agree")
    if partner is not None and args.stride_average:
        ap.error("--stride-average is the single physics fly's: in a pair each fly's senses see its body as it is")
    if args.social is not None:
        try:
            social = SocialConfig.from_list(args.social)
        except ValueError as e:
            ap.error(str(e))
    else:
        social = SocialConfig.from_list(None)
    if args.physics_levers is not None and args.body != "physics":
        ap.error("--physics-levers only applies to the physics body: add --body physics")
    levers = None                                # None: the game's default (none for one fly; the pair's adopted set)
    if args.physics_levers is not None:
        from .physics import parse_levers
        try:
            levers = parse_levers(args.physics_levers)
        except ValueError as e:
            ap.error(str(e))
    if args.body == "physics":
        from .physics import available, unavailable_reason
        if not available():
            raise SystemExit(unavailable_reason())

    profile = "pure" if args.pure else args.profile
    if args.stride_average and args.body != "physics":
        ap.error("--stride-average only applies to the physics body: add --body physics")
    if args.backend == "numba":                  # said before the data is loaded, not as a traceback after
        from .fastbrain import available as numba_available
        if not numba_available():
            ap.error("--backend numba needs the numba package: pip install numba (or leave out --backend to use NumPy)")
    if args.backend == "cupy":                   # the same for the GPU: what is missing, in one line, before loading
        from .gpubrain import unavailable_reason
        reason = unavailable_reason()
        if reason is not None:
            ap.error(f"--backend cupy: {reason} (or leave out --backend to use the CPU)")
        if args.brain_procs == "on":
            ap.error("--brain-procs on cannot hold a GPU brain: with --backend cupy one brain process serves every fly "
                     "(leave --brain-procs at auto), or use --brain-procs off to keep it in this process")
    overrides = {"seed": args.seed, "dt": 1.0 if args.fast else args.dt, "backend": args.backend}
    if args.fatigue is not None:
        overrides["fatigue_mv"] = args.fatigue
    if args.kenyon_gain is not None:
        overrides["kenyon_gain"] = args.kenyon_gain
    if args.noise:
        try:
            hz, mv = (float(x) for x in args.noise.split(":"))
        except ValueError:
            ap.error(f"--noise wants HZ:MV, e.g. 5:15 (got {args.noise!r})")
        overrides.update(noise_hz=hz, noise_mv=mv)
    from .parts import PartsList
    parts_list = PartsList(curated=args.curated or "modulators")
    if args.parts:
        overrides["parts"] = parts_list
    print("Loading the fly's nervous system...", file=sys.stderr)
    conn = load_connectome(female=args.female)
    # with a GPU brain in the brain process (6.5) the copy built here only feeds the process its settings and the parts
    # counts line below: it is built on the CPU, so this process opens no CUDA context of its own
    gpu_here = args.backend == "cupy" and args.brain_procs == "off"
    brain = build_brain(conn, profile, **{**overrides, "backend": args.backend if gpu_here or args.backend != "cupy" else "auto"})
    if args.backend == "cupy":
        pass                                     # said once the game exists, from the brain process's own settings
    elif brain.backend == "numba":
        print("Brain integrator: compiled (numba).", file=sys.stderr)
    elif args.backend == "numpy":
        print("Brain integrator: NumPy (as asked with --backend numpy).", file=sys.stderr)
    else:
        print("Brain integrator: NumPy. For a brain about twice as fast (same spikes): pip install numba", file=sys.stderr)
    if args.no_learning and brain.plasticity is not None:
        brain.plasticity.enabled = False
    if brain.parts is not None:
        c = brain.parts.counts
        print(f"Parts list: on ({c['modulatory_neurons']:,} modulatory neurons, {c['co_release_neurons']:,} of them also keeping "
              f"their fast synapses; {c['graded_neurons']:,} graded cells).", file=sys.stderr)
    if args.body == "physics" and levers:
        print("Physics levers on (each changes the physics; docs/SCIENCE.md 13.5): " + ", ".join(levers) + ".", file=sys.stderr)
    elif args.body == "physics" and levers is None:
        from .physics import DEFAULT_LEVERS
        print("Physics levers on, the adopted default (docs/SCIENCE.md 13.5): " + ", ".join(DEFAULT_LEVERS)
              + " (it changes no number; --physics-levers none switches it off).", file=sys.stderr)
    if args.body == "physics" and partner is not None:
        print("Body: physics for both flies (NeuroMechFly v2 in one MuJoCo world, each with its own brain, able to touch; "
              "slower than real time, about a tenth with two flies; both drawn at real size).", file=sys.stderr)
    elif args.body == "physics":
        print("Body: physics (NeuroMechFly v2 legs in MuJoCo; about a tenth of real time"
              + ("; the senses see the pose averaged over a stride)." if args.stride_average else ")."), file=sys.stderr)
    partner_spec = None
    if partner is not None:
        print("Loading the partner's nervous system...", file=sys.stderr)
        pconn = load_connectome(female=(partner == "female"))
        partner_spec = {"conn": pconn, "brain_kwargs": {k: v for k, v in overrides.items() if k != "parts"},
                        "parts": parts_list if args.parts else False, "autopilot": not args.no_autopilot}
    game = Game(brain, autopilot=not args.no_autopilot, seed=args.seed, columnar=not args.no_columnar,
                profile_name=profile, brain_factory=lambda c, **kw: build_brain(c, profile, **{**overrides, **kw}),
                parts_list=parts_list, brain_kwargs={k: v for k, v in overrides.items() if k != "parts"}, body=args.body,
                stride_average=args.stride_average, brain_procs=args.brain_procs, partner=partner_spec, social=social,
                physics_levers=levers)
    del brain                                    # the game owns it now (or, in its own process, has let it go)
    if args.backend == "cupy":
        backend = game.flies[0].io.settings().get("backend")
        where = "in the brain process" if game.brain_mode == "server" else "in this process"
        if backend == "cupy":
            print(f"Brain integrator: the GPU (CuPy) {where}, as asked with --backend cupy.", file=sys.stderr)
        else:
            print(f"Brain integrator: {backend} {where} (the GPU was asked for with --backend cupy).", file=sys.stderr)
    if partner_spec is not None:
        where = {"server": "in the one brain process", "procs": "in its own process"}.get(game.brain_mode, "in this process")
        print(f"Partner: {pconn.dataset} ({pconn.sex}), its brain {where}, as is the first fly's; "
              f"the channels between them: {', '.join(social.names()) or 'none'} (hand-built, see What's real here?).",
              file=sys.stderr)
    if args.no_learning:
        game.learning_on = False
    if args.grow:
        r = game.action({"type": "grow", "level": args.grow, "seed": args.grow_seed})
        if not r["ok"]:
            raise SystemExit(r["error"])
        print(f"Growing a fly from its {args.grow} wiring rules (seed {args.grow_seed}) in the background...", file=sys.stderr)
    serve(game, port=args.port, open_browser=not args.no_browser, host=args.host)


if __name__ == "__main__":
    sys.argv[0] = "python -m virtual_fly.play"     # argparse names the program after it (else "play.py", before Python 3.14)
    main()
