# Architecture

```
fly_brain.py / fly_game.py / my_first_fly.py     thin entry points (kept from the starter kit)
virtual_fly/
  connectome.py    the data: FLYB reader, population specs, input/output indices, cell-type graph
  brain.py         the simulation: LIF network, optional brakes/noise/modulation, monitors, checkpoints
  fastbrain.py     the same integration step as compiled numba kernels (optional, same spikes, ~2x faster)
  plasticity.py    mushroom-body learning: dopamine-gated depression of KC->MBON synapses
  pathways.py      static analysis: strongest routes between populations, lesion candidates
  experiments.py   validated protocols, seeds, sweeps, lesion scans, JSON export
  genetics.py      gene-expression populations (fru, dsx, transmitter genes), FlyBase links, NeuronBridge lookups
  wiring.py        the genome as a recipe: cell-type wiring rules, grown flies, the rank bottleneck
  parts.py         the genes as a parts list: slow modulators (DA/OA/5-HT), graded cell types, per-type thresholds,
                   curated transmitters and receptor signs (both from vfb.py), receptor facts from the literature,
                   neurons that release locally (APL)
  vfb.py           the anatomy-ontology join (Virtual Fly Brain): fbbt:/rx: selectors, per-type facts, curated
                   transmitters, receptor expression per class; reads data/fbbt_*.json.gz and data/vfb_receptors.json.gz
tools/
  build_vfb_data.py    fbbt.obo (+ the connector overlay) -> data/fbbt_map.json.gz, data/fbbt_tree.json.gz
  merge_vfb_harvest.py a Virtual Fly Brain connector harvest -> tools/vfb_overlay.json, data/vfb_receptors.json.gz
  settings.py      named model profiles (pure / game / brakes)
  cli.py           `python fly_brain.py ...`
  world.py         the arena: food, posts, odour sources and plume puffs, wind, the drum, the female
  body.py          the fly's body: inertia, gait, appendages, collisions
  senses/
    vision.py      retina (two compound eyes), feature detectors, columnar T4/T5 motion detectors
    olfaction.py   odours -> glomeruli -> ORN rates, with adaptation
    taste.py       mouth and foreleg taste
    mechano.py     wind, sound, touch, dust, proprioception
  game.py          the sensorimotor loop: senses -> brain -> decoder -> body; internal state; events
  scenarios.py     scripted protocols (conditioning, courtship, plume, escape)
  server.py        HTTP + Server-Sent Events API (docs/API.md)
  play.py          `python fly_game.py ...`
  web/             the browser page (no build step, no dependencies)
tests/             pytest suite on a small synthetic connectome (no download needed)
docs/              API.md (server contract), SCIENCE.md (what was measured and why), this file
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
6. **Behaviour selection** (`Game.choose_mode`): giant-fibre spikes win outright (a jump);
   otherwise the strongest drive above its threshold wins, with hysteresis.
7. **The body** moves with inertia, respecting the wall and posts; appendages follow their drives.
8. **The world** steps: puffs drift and grow, the female walks, the drum turns.
9. **State** is serialised to JSON and pushed to every browser subscriber.

## Threads

The game loop runs in one thread; the HTTP server handles requests in others. Anything that
changes the wiring (silence, modulate, forget) takes `brain.lock`; reads of `state_json` are
atomic swaps of an immutable bytes object; actions are queued and applied at the start of a tick.

## Performance notes

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
