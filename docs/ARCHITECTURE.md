# Architecture

```
fly_brain.py / fly_game.py / my_first_fly.py     thin entry points (kept from the starter kit)
virtual_fly/
  connectome.py    the data: FLYB reader, population specs (and a file's name aliases), input/output indices, cell-type graph
  flywire.py       the female fly: builds FlyWire 783 in the FLYB format from its public sources, with the table of
                   the kit's MaleCNS names for FlyWire's cells (load_connectome(female=True))
  brain.py         the simulation: LIF network, optional brakes/noise/modulation, monitors, checkpoints
  fastbrain.py     the same integration step as compiled numba kernels (optional, same spikes, ~2x faster)
  gpubrain.py      the same step as CUDA kernels through CuPy (optional, --backend cupy, same spikes to the last bit): the
                   state lives on the GPU, chunks of steps run as one captured graph, the host replays its bookkeeping from
                   the spike log; GpuFlyBrain is the FlyBrain subclass that FlyBrain(backend="cupy") builds
  plasticity.py    mushroom-body learning: dopamine-gated depression of KC->MBON synapses
  pathways.py      static analysis: strongest routes between populations, lesion candidates
  experiments.py   validated protocols, seeds, sweeps, lesion scans, JSON export
  retest.py        the game's background re-test (the survival report) in a separate low-priority process
  genetics.py      gene-expression populations (fru, dsx, transmitter genes), FlyBase links, NeuronBridge lookups
  wiring.py        the genome as a recipe: cell-type wiring rules, grown flies, the rank bottleneck
  parts.py         the genes as a parts list: slow modulators (DA/OA/5-HT), graded cell types, per-type thresholds,
                   curated transmitters and receptor signs (both from vfb.py), receptor facts from the literature,
                   neurons that release locally (APL)
  vfb.py           the anatomy-ontology join (Virtual Fly Brain): fbbt:/rx: selectors, per-type facts, curated
                   transmitters, receptor expression per class; reads data/fbbt_*.json.gz and data/vfb_receptors.json.gz
  settings.py      named model profiles (pure / game / brakes)
  cli.py           `python fly_brain.py ...`
  __main__.py      `python -m virtual_fly ...`, the same command line
  world.py         the arena: food, posts, odour sources and plume puffs, wind, the drum, the female
  body.py          the fly's body: inertia, gait, appendages, collisions
  physics.py       the optional physics body: NeuroMechFly v2 legs in MuJoCo through flygym 2.1 (--body physics; about
                   real time): the fly built from flygym's parts, the kit's wall, the recorded stride and the CPG controller
  physics_pair.py  two (or more) physics bodies in ONE MuJoCo world, able to touch (--partner ... --body physics; the
                   two-flies work): the shared world, each fly's body in it, the contact pairs and the physical tap
  senses/
    vision.py      retina (two compound eyes), feature detectors, columnar T4/T5 motion detectors
    olfaction.py   odours -> glomeruli -> ORN rates, with adaptation
    taste.py       mouth and foreleg taste
    mechano.py     wind, sound, touch, dust, proprioception
    social.py      how one fly reaches another (Phase 1 of the two-flies work): seen, song, contact, collide, cVA,
                   mating status, body touch; every channel a labelled hand-built encoder with a switch
  agent.py         one fly: its brain handle, senses, decoder, body and bookkeeping (FlyAgent); the game holds the dish
  brainio.py       the seam between a fly and its brain: LocalBrain (in this process), ProcessBrain (a child process,
                   one per brain when there are two flies) or a GpuBrainServer handle (one child process owning every
                   brain, where the GPU brains live), the same spikes either way; advance_all is the lockstep
  game.py          the sensorimotor loop over every fly: senses -> brains (in lockstep) -> decoders -> bodies; the world's
                   clock; internal state; events; the actions from the browser
  scenarios.py     scripted protocols (conditioning, courtship, plume, escape; with a partner, the two-fly courtship)
  server.py        HTTP + Server-Sent Events API (docs/API.md); every per-fly endpoint takes ?fly=k, actions a "fly" field
  play.py          `python fly_game.py ...`
  web/             the browser page (no build step; one vendored library, three.js (MIT), loaded only for the 3-D view); with a partner it shows both flies, a fly menu
                   arena3d.js is the 3-D dish (three.js; the legs animated from the gait phase, labelled "not physics"),
                   models/ the NeuroMechFly meshes as glTF and the gait atlas (tools/build_fly_model.py), vendor/three/ the library
                   (`#focusSel`) picks which one the panels, retina and brain map follow, per-fly actions carry that
                   fly's id (`setActionFly` in util.js), and two brain maps stay alive so no WebGL context is rebuilt
tools/
  build_vfb_data.py    fbbt.obo (+ the connector overlay) -> data/fbbt_map.json.gz, data/fbbt_tree.json.gz
  merge_vfb_harvest.py a Virtual Fly Brain connector harvest -> tools/vfb_overlay.json, data/vfb_receptors.json.gz
  harvest_neuprint_rois.py  neuPrint (token; run by .github/workflows/neuprint-harvest.yml) -> data/mb_roi_connectivity.json.gz,
                       APL's and DPM's connections split by region
  vfb_overlay.json     the connector harvest: FBbt classes for the names the OBO lacks (an input of build_vfb_data.py)
  bench_two_flies.py   baseline speeds for the two-flies work: one brain, the male and female brains side by side,
                       game ticks with the drawn and the physics body, and the two-fly game with both as physics
                       bodies per contact set (hand-run; docs/TWO_FLIES_PLAN.md 4.8, 8.2)
  physics_table.py     docs/SCIENCE.md 6.7's physics table measured again on the kit as it is: the gate every speed
                       lever of the two-flies work is checked against (hand-run; docs/TWO_FLIES_PLAN.md 8.2)
  render_replay.py     a saved physics recording (recordings/<id>, recording.py) rendered to an MP4 at real speed by
                       MuJoCo's renderer: overhead, follow or side camera, the joints interpolated between ticks
                       (hand-run; docs/TWO_FLIES_PLAN.md 8.9)
  refit_stride.py      NeuroMechFly's recorded stride (the 3-D view's gait atlas, from flygym 1.2.1) refitted onto flygym
                       2.1's skeleton, with every atlas geom's offset on the 2.1 bodies: web/models/nmf_stride.npz, the
                       physics body's step tables and the replays' pose mapping (hand-run once; --check verifies)
  alias_audit.py       every population spec the kit's code names (virtual_fly/specs.py), counted on the male, FlyWire
                       and BANC files term by term, so that a union with one empty member cannot hide behind the others'
                       cells; an empty term on BANC comes with candidate names (hand-run; docs/TWO_FLIES_PLAN.md 9.5)
  golden_hashes.py     what one fly does on the real data, byte for byte: --save before a change, --compare after
                       (tests/test_golden_single_fly.py is the same check on the synthetic connectome, in CI)
  compare_experiments.py  every difference between two `fly_brain.py --json` files (a later run against the baselines)
data/              the connectome (downloaded the first time; the female fly's is built there) and the small tables
                   read by vfb.py and parts.py (fbbt_*.json.gz, vfb_receptors.json.gz, mb_roi_connectivity.json.gz);
                   an installed copy downloads and builds into ~/.cache/virtual-fly instead, and FLY_DATA_DIR,
                   when set, replaces either folder
tests/             pytest suite on a small synthetic connectome (no download needed)
docs/              API.md (server contract), SCIENCE.md (what was measured and why), this file;
                   TWO_FLIES_PLAN.md and TWO_FLIES_PROGRESS.md (the two-flies work: the plan and its log)
```

## The loop, one tick at a time

Every tick is 25 ms of fly time:

1. **Actions** from the browser are applied (drops, zaps, tool changes, scenario steps).
2. **Senses** (`Game.senses`): the retina is re-rendered from the fly's pose and the visible
   objects; feature detectors and the columnar motion detectors turn it into rates for real visual
   neuron types; the nose samples the plume at both antennae; the mouth and forelegs check what
   they touch; the antennae report wind and sound; bristles report touch. The result is a map
   `population spec -> Hz` plus per-neuron rates for the T4/T5 columns, and a small dict of what the
   fly felt, for the screen.
3. **The brain** (`FlyBrain.step` x 50): every neuron in the connectome is integrated (a brain
   that is completely at rest skips the maths), stimulated sensory neurons fire as Poisson processes, spikes
   propagate over the 6.3 million connections with the 1.8 ms delay, and the plasticity rule updates
   the KC->MBON weights from the KC and dopamine traces.
4. **Readouts**: firing rates of the key neuron populations over the tick; silenced neurons count
   for the display but not for the body.
5. **Decoder** (`MotorDecoder.decode`): descending-neuron rates -> smoothed motor drives (forward,
   yaw, backward, halt, feed, groom, song, court).
6. **Behaviour selection** (`Game.choose_mode`): a giant-fibre burst wins outright (a jump);
   otherwise the strongest drive above its threshold wins, with hysteresis.
7. **The body** moves with inertia, respecting the wall and posts; appendages follow their drives.
8. **The world** steps: puffs drift and grow, the female walks, the drum turns.
9. **State** is serialised to JSON and pushed to every browser subscriber.

## Threads and processes

The game loop runs in one thread; the HTTP server handles requests in others. Anything that
changes the wiring (silence, modulate, forget) goes through the fly's `BrainIO` seam: with the brain
in this process that takes `brain.lock`; with the brain in a child process every request-and-reply
pair holds the brain's own lock, so the game thread's tick and an HTTP thread's query never
interleave on the pipe. Reads of `state_json` are atomic swaps of an immutable bytes object; actions
are checked on arrival (a bad one is refused with its reason), queued, and applied at the start of a tick.

With `--partner` each brain runs in its own child process (`spawn`, as the re-test child does; the
parent warms the numba cache first so the children load it). A tick sends every brain its rates,
then waits for every `BrainTick` (the lockstep), then moves the bodies; a child that dies or stops
answering is reported and stopped, never waited on, and the other brain's lock is released. After
`Ctrl+C`, `server.serve` stops and joins the loop thread and only then `Game.close()` closes the
children (the order that keeps a MuJoCo body from being torn down mid-step). The re-test of a
rebuilt brain is a third child, at low priority, as before.

With `--backend cupy` both brains live in **one** child process, the brain server (`GpuBrainServer`
in `brainio.py`): one CUDA context, launched for both brains back to back, away from the web
server, so a CUDA failure stops that child and is reported like a dead brain process rather than
taking the page down. The game process keeps a CPU-built copy of the first fly only long enough to
describe it to the child; the re-test child always runs on the CPU (its answer is about the wiring,
and the CPU is bit-identical); a rebuild (grow, parts) happens inside the server, which then swaps
the new GPU brain in. `--brain-procs off` with `--backend cupy` keeps the GPU brain in the game
process instead (the way `fly_brain.py` runs it).

## Performance notes

* The GPU backend (`gpubrain.py`) runs the same step as hand-written CUDA kernels: the dense pass
  over every neuron, then an event-driven propagation (spiking neurons mark their outgoing edges
  in a bitmap; every hit target walks its incoming edges in presynaptic order and adds the marked
  kicks one by one, from the queue slot's value), so each target's additions happen in the CPU's
  order and no float atomics decide a result. Ten steps (five at `dt` 1.0) form one chunk captured
  as a CUDA graph; the host draws the chunk's random numbers ahead in the CPU's order, uploads them
  with the staged tone gain and any changed weights, launches, downloads the spike log, sorts each
  step's list and replays the host-side work. Measured (docs/SCIENCE.md 11.3): 61 µs per step for the busy male
  against numba's 131 on one core, 69 against 108 for the female; about 1.9 ms of device time per 25 ms tick
  and 1.3-2.3 ms of host work. In the brain server the two brains' ticks are generators resumed in turn
  (a chunk launched, the other brain's chunk launched, the first collected and replayed while the second
  runs), so the two-fly game holds 3.6x real time on the GPU against 2.8x with the CPU's two processes.
* `FlyBrain.step` has two interchangeable integrators that produce the same spikes to the last
  one (the test suite checks this on every optional mechanism, and `--backend` picks one). The
  NumPy one is the starter kit's dense loop: a handful of passes over the 176k-element state arrays
  per 0.5 ms step, plus `np.add.at` to scatter the spikes' synaptic kicks. The compiled one
  (`fastbrain.py`, used automatically when `numba` is installed) does the same float32 arithmetic in
  the same order in one fused, vectorised pass and a plain loop over the spiking neurons' outgoing
  connections: 0.28 ms per step against 0.63 ms, all on one core. Two variants were measured and
  rejected: an active-set integrator that touched only non-resting neurons (slower in NumPy: with any
  stimulus on, a third to a half of the brain is slightly off rest, so the gathers cost more than the
  dense passes they save), and a multi-threaded kernel (1.3x faster on an idle machine, ten times
  *slower* the moment one other process such as the browser used a core, because every 50 µs
  parallel region waited for a descheduled worker). One guard the starter lacks: every 20 steps,
  voltages and synaptic inputs that have decayed below a nanovolt are snapped to zero, because
  float32 values drifting into the denormal range slow every array operation several-fold (a busy,
  never-quiet game brain ran at half speed before this guard). A brain that is completely at rest
  skips the maths entirely; `--fast` (a 1 ms step) halves the cost with every classic experiment
  still in range. On the 4-core machine this was built on, the game with a busy brain (sugar, a
  female, the drum) runs at about 1.5x real time compiled and 0.7x in NumPy, and 2.7x / 1.2x with
  `--fast`.
* The connectome's input index (`col_ptr`), presynaptic array and cell-type graph are built lazily
  on first use (a second or two each).
* The layout JSON (2.4 MB) is built once; large responses are gzip-compressed.
