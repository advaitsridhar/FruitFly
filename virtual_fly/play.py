"""
Start the game: ``python fly_game.py`` or ``python -m virtual_fly.play``.

    --port 9000          use another port
    --host 0.0.0.0       let other computers on your network open the game (anyone on it can drive the fly)
    --no-browser         don't open a browser tab
    --profile pure       the paper's model exactly (expect runaway loops after bitter, dust and smells)
    --no-autopilot       start with the hand-built walking urge switched off
    --no-learning        start with mushroom-body plasticity switched off
    --no-columnar        don't drive the connectome's own T4/T5 motion-detector columns from the retina
    --noise 2:1          background kicks per neuron per second : size in mV
    --fast               brain time step 1 ms instead of 0.5 ms (about twice as fast; all six
                         classic experiments still pass)
    --parts              start with the genes as each neuron's parts list (the Genome card toggles it)
    --body physics       walk with NeuroMechFly v2 legs in MuJoCo instead of the drawn body (needs flygym;
                         runs at about a tenth of real time)
"""

from __future__ import annotations

import argparse
import sys

from .connectome import load_connectome
from .game import Game
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
    ap.add_argument("--noise", metavar="HZ:MV", default=None, help="background kicks, e.g. 2:1.0 (0 = off)")
    ap.add_argument("--kenyon-gain", type=float, default=None)
    ap.add_argument("--no-autopilot", action="store_true", help="start with the hand-built walking urge switched off")
    ap.add_argument("--no-learning", action="store_true", help="switch mushroom-body plasticity off")
    ap.add_argument("--no-columnar", action="store_true",
                    help="don't drive the connectome's T4/T5 motion-detector columns from the retina")
    ap.add_argument("--dt", type=float, default=0.5, help="brain time step in ms (0.5 default; 1.0 = twice as fast, slightly coarser)")
    ap.add_argument("--fast", action="store_true", help="same as --dt 1.0: for computers that run the brain below real time")
    ap.add_argument("--backend", choices=("auto", "numpy", "numba"), default="auto",
                    help="brain integrator: the compiled numba kernels when numba is installed (auto), or plain NumPy")
    ap.add_argument("--grow", metavar="LEVEL", default=None,
                    help="start with a fly grown from its wiring rules: type, class or bottleneck:K (the Genome card does the same)")
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
    args = ap.parse_args(argv)
    if args.body == "physics":
        from .physics import available, unavailable_reason
        if not available():
            raise SystemExit(unavailable_reason())

    profile = "pure" if args.pure else args.profile
    if args.stride_average and args.body != "physics":
        ap.error("--stride-average only applies to the physics body: add --body physics")
    overrides = {"seed": args.seed, "dt": 1.0 if args.fast else args.dt, "backend": args.backend}
    if args.fatigue is not None:
        overrides["fatigue_mv"] = args.fatigue
    if args.kenyon_gain is not None:
        overrides["kenyon_gain"] = args.kenyon_gain
    if args.noise:
        hz, mv = (float(x) for x in args.noise.split(":"))
        overrides.update(noise_hz=hz, noise_mv=mv)
    from .parts import PartsList
    parts_list = PartsList(curated=args.curated or "modulators")
    if args.parts:
        overrides["parts"] = parts_list
    print("Loading the fly's nervous system...", file=sys.stderr)
    conn = load_connectome(female=args.female)
    brain = build_brain(conn, profile, **overrides)
    if brain.backend == "numba":
        print("Brain integrator: compiled (numba).", file=sys.stderr)
    elif args.backend == "numpy":
        print("Brain integrator: NumPy (as asked with --backend numpy).", file=sys.stderr)
    else:
        print("Brain integrator: NumPy. For a brain about twice as fast (same spikes): pip install numba", file=sys.stderr)
    if args.no_learning and brain.plasticity is not None:
        brain.plasticity.enabled = False
    if brain.parts is not None:
        c = brain.parts.counts
        print(f"Parts list: on ({c['modulatory_neurons']:,} modulatory neurons, {c['graded_neurons']:,} graded cells).", file=sys.stderr)
    if args.body == "physics":
        print("Body: physics (NeuroMechFly v2 legs in MuJoCo; about a tenth of real time"
              + ("; the senses see the pose averaged over a stride)." if args.stride_average else ")."), file=sys.stderr)
    game = Game(brain, autopilot=not args.no_autopilot, seed=args.seed, columnar=not args.no_columnar,
                profile_name=profile, brain_factory=lambda c, **kw: build_brain(c, profile, **{**overrides, **kw}),
                parts_list=parts_list, brain_kwargs={k: v for k, v in overrides.items() if k != "parts"}, body=args.body, stride_average=args.stride_average)
    if args.no_learning:
        game.learning_on = False
    if args.grow:
        r = game.action({"type": "grow", "level": args.grow, "seed": args.grow_seed})
        if not r["ok"]:
            raise SystemExit(r["error"])
        print(f"Growing a fly from its {args.grow} wiring rules (seed {args.grow_seed}) in the background...", file=sys.stderr)
    serve(game, port=args.port, open_browser=not args.no_browser, host=args.host)


if __name__ == "__main__":
    main()
