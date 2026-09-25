# The science behind the Virtual Fly kit

This document says what the kit simulates, what was measured to decide how to build it, and where
each number comes from. It is written in the same spirit as the starter kit's README: the wiring
diagram is real, the neuron model is the simplest one that works, and every part that somebody
designed by hand is labelled **hand-built**.

Conventions used throughout:

* **pure** = the published model (Shiu et al. 2024) with the male-connectome calibration of the
  fly-brain-minecraft project: nothing added, nothing silenced.
* **probe-game** = the configuration the probe experiments called "game": fatigue 0.05 mV/spike
  plus `silence("class:ALLN")`, run on the starter kit's `fly_brain.py` (kenyon_gain 0.25 unless
  stated, dopamine neurons *not* silenced). The probes were run before the kit's final `game`
  profile existed.
* **game** = the profile the game runs today (section 1.6): fatigue 0.05 mV/spike, `class:ALLN`
  and `class:DAN` silenced, kenyon_gain 1.0, mushroom-body plasticity on.
* Unless stated otherwise a measurement is: `dt` 0.5 ms, seed 0, a 600 ms Poisson stimulus of
  which the first 100 ms are discarded, rates = mean spikes/s per neuron over the remaining
  500 ms, "after" = whole-brain spikes/s measured 500-1000 ms after the stimulus stops.
* Rates in this model are only meaningful relative to each other. Every neuron is identical, so
  "MN9 fires at 47 Hz" means "the sugar pathway reaches MN9", not that a real MN9 fires at 47 Hz.

The probe scripts and their raw outputs are referred to by directory name (`olfaction-learning`,
`synaptic-depression`, `computed-vision`, `courtship-mechano`); the two profile runs quoted in
section 2 were made with `python fly_brain.py --profile <name>` on this machine.

---

## 1. The model

### 1.1 The wiring diagram

| item | value | source |
|---|---|---|
| dataset | MaleCNS v1.0 (`male-cns:v1.0` on neuPrint), the complete male *Drosophila* central nervous system | Berg et al. 2026 |
| neurons in the file | 176,422 | loader |
| connections kept | 6,287,749 (every pair with 5 or more synapses) | loader |
| synapses in those connections | 90,296,905 | loader |
| sign of a neuron | from its predicted transmitter: acetylcholine, dopamine, octopamine, serotonin = +1; GABA, glutamate = −1 | file; checked in `olfaction-learning/q4b` |
| medulla column coordinates (`hex1`, `hex2`) | 23,720 neurons of 15 columnar types (L1 L2 L5 C3 Mi1 Mi4 Mi9 T1 Tm1 Tm2 Tm9 Tm20 in both eyes; L3 C2 Tm4 right eye only); none for T4/T5 | `computed-vision/step1` |
| file | `malecns-v1.0.flyb.gz`, built by fly-brain-minecraft from neuPrint, pinned to one commit and checked by SHA-256 | `connectome.py` |

Two consequences of the sign rule matter later: **dopamine is excitatory** (so a dopamine neuron
drives whatever it synapses onto, exactly like a cholinergic neuron) and **glutamate is
inhibitory** (so the glutamatergic mushroom-body output neurons can only ever silence things).

### 1.2 The neuron

Every neuron is the same leaky integrate-and-fire unit (Shiu et al. 2024). With `v` the voltage
and `g` the synaptic input, both in mV above rest:

```
tau_m dv/dt = -v + g          tau_m = 20 ms
tau_s dg/dt = -g              tau_s = 5 ms
spike when v >= 7 mV  ->  v = g = 0, refractory 2.2 ms
1.8 ms after a spike, every target j receives  g_j += sign_i * n_syn(i,j) * 0.275 mV * gain
```

| parameter | value | note |
|---|---|---|
| `MV_PER_SYNAPSE` | 0.275 mV | Shiu et al. 2024 |
| `gain` | 0.65 | fly-brain-minecraft calibration for the male brain (section 1.3) |
| `THETA` | 7 mV above rest (−45 mV) | Shiu et al. 2024 |
| `TAU_M`, `TAU_S` | 20 ms, 5 ms | Shiu et al. 2024 |
| refractory, delay | 2.2 ms, 1.8 ms | Shiu et al. 2024 |
| `dt` | 0.5 ms (the paper's Brian2 runs used 0.1 ms) | `--dt 0.1` reproduces the paper's step, 5x slower |
| integration | exact for the two linear equations over one step (`coupling` = tau_s/(tau_m−tau_s) · (e^{−dt/tau_m} − e^{−dt/tau_s})) | `brain.py` |
| refractory neurons | frozen at reset; input arriving meanwhile accumulates in `g` | `brain.py` |

Stimulation is a Poisson clamp: a neuron in a stimulated population fires in each step with
probability `hz · dt / 1000`; a neuron that belongs to two stimulated populations keeps the higher
rate. This is how Shiu et al. drive their sensory neurons, and how every sense in the game reaches
the brain.

### 1.3 What "gain 0.65" and "Kenyon gain" mean

* **gain** multiplies every synaptic weight. With gain 0.65 one spike across a connection of
  `k` synapses adds `k · 0.275 · 0.65 = 0.179 k` mV to the target's `g`. At steady state
  `v = g = rate · w · tau_s`, so a single 100 Hz input needs about 78 synapses to bring its
  target to threshold on its own (`courtship-mechano/task1b`). The value 0.65 comes from
  fly-brain-minecraft: the male connectome has more synapses per neuron than the female brain
  the 0.275 mV figure was tuned on, and the literal value over-excites it.
* **Kenyon gain** (`kenyon_gain`) multiplies every synaptic weight whose *target* is a Kenyon
  cell (KC), the ~4,000 intrinsic neurons of the mushroom body. Their outputs are not scaled.
  fly-brain-minecraft and the pure profile use 0.25 to keep the KC code sparse; the game uses 1.0
  because, once the antennal lobe is calmed (section 3), 0.25 leaves the KCs silent (section 4.3).

### 1.4 Integration (what the game actually runs)

`FlyBrain.step()` is the starter kit's dense loop: every 0.5 ms it decays `v` and `g` for all
176,422 neurons in a few NumPy passes, delivers the kicks queued 1.8 ms earlier, finds the
neurons above threshold, and scatters their kicks into the delay queue with `np.add.at`. On this
machine the six classic experiments run in the same wall time as in the starter kit and produce
the same spikes to the last one (30,740 / 87,216 / 131,068 spikes for sugar / looming / dust in
both). An active-set variant that integrated only neurons off rest was built and measured slower
on every experiment: with a stimulus on, 38-52 % of the brain sits slightly off rest, and the
gathers and scatters cost more than the dense passes they save.

* **The compiled integrator** (`fastbrain.py`, since v2.1; used when `numba` is installed,
  `--backend numpy` gives the loop above): the same float32 operations in the same order (leak,
  integrate, freeze the refractory neurons, flush, threshold, union with the forced Poisson spikes,
  then the kicks connection by connection in `np.add.at`'s order), fused into one vectorised pass
  and one loop, so it is bit-identical: identical spike trains on the real data for the classic
  stimuli (1,000 recorded steps compared), and on the synthetic connectome with every mechanism
  (fatigue, depression, jitter, noise, learning, `dt` 1.0) in `tests/test_fastbrain.py`. Per 0.5 ms
  step with sugar and looming on: NumPy 0.63 ms, compiled 0.28 ms (0.11 ms dense pass, 0.04 ms
  scan for threshold crossings, 0.06 ms scatter of ~2,700 kicks, 0.03 ms Python), on one core.
  A multi-threaded version of the dense pass was 1.3x faster on an idle machine and ten times
  slower with one other busy process on the box, so the kernel is single-threaded on purpose.
  The busy game tick (sugar, female, drum) went from 34 to 17 ms (1.5x real time) at `dt` 0.5 and
  from 20 to 9 ms (2.7x) at `dt` 1.0.
* Every 20 steps, `v` and `g` values below 1 nV (`FLUSH_MV` = 1e-6 mV, seven million times below
  threshold) are snapped to 0. Without this, values decaying for hundreds of milliseconds drift into the float32 denormal
  range and the CPU slows every array operation several-fold: a busy game brain went from 76 ms
  to 36 ms per 25 ms tick when the guard was added. The classic experiments are unchanged to the
  spike.
* Every 200 steps (100 ms) the brain checks whether it is *quiet*: no stimulus, no noise, no
  pending delayed input, nothing refractory, and `|v|`, `|g|` below 0.01 mV everywhere. A quiet
  brain skips the maths entirely until input arrives; fatigue keeps fading analytically meanwhile,
  and the plasticity traces keep decaying.
* Fatigue (section 1.5) is faded in blocks of 20 steps rather than every step.
* The game simulates 25 ms of brain time per world tick (`TICK_MS`, 50 steps) at 40 ticks per
  second, and slows the world down (the "slow motion" notice) when the brain is busier than real
  time allows. `--fast` uses a 1 ms step (25 per tick): about twice as fast, and all six classic
  experiments stay in range (MN9 48 Hz, giant fibre 307 Hz, aDN1 140 Hz in the game profile).

### 1.5 Optional mechanisms (all off in `FlyBrain(conn)`)

| mechanism | what it does | parameters | literature |
|---|---|---|---|
| spike-frequency adaptation ("fatigue") | each spike raises that neuron's threshold by `fatigue_mv`; the increase decays with `fatigue_ms` | game: 0.05 mV, 2000 ms | generic; a stand-in for the many adaptation currents real neurons have |
| short-term synaptic depression | each spike uses a fraction `U` of the neuron's releasable resource `x`; `x` recovers with time constant `tau_rec`; every outgoing weight is multiplied by `x` at the moment of the spike | brakes: U 0.1, tau 100 ms | Tsodyks & Markram 1997 |
| background noise | independent Poisson kicks of `noise_mv` at `noise_hz` per neuron | off in every profile (`--noise HZ:MV` switches it on) | section 3.4 |
| threshold jitter | fixed per-neuron threshold offset, sd `threshold_jitter` mV, seeded | off | generic heterogeneity |
| output modulation | `silence(spec)` zeroes a population's outgoing weights (its neurons still fire, like tetanus-toxin expression); `modulate(spec, f)` scales them | game silences `class:ALLN` and `class:DAN`; hunger modulates the sugar neurons' *input* rates instead (hand-built, `taste.py`) | Shiu et al. 2024 use silencing the same way |
| mushroom-body plasticity | dopamine-gated depression of KC→MBON synapses | section 4.7 | Hige et al. 2015; Handler et al. 2019 |
| recording, monitors, checkpoints | every spike; per-population rate histories; full-state snapshots to branch an experiment | | |

### 1.6 Profiles

| profile | fatigue | silenced | kenyon_gain | short-term depression | noise | plasticity | purpose |
|---|---|---|---|---|---|---|---|
| `pure` | 0 | nothing | 0.25 | off | off | off | Shiu et al. 2024 exactly (with gain 0.65) |
| `game` | 0.05 mV/spike | `class:ALLN` (420 antennal-lobe local neurons), `class:DAN` (340 dopamine neurons) | 1.0 | off | off (`--noise HZ:MV` switches it on) | on | what the game runs |
| `brakes` | 0.05 mV/spike | `class:DAN` | 1.0 | U = 0.1, tau_rec = 100 ms | off | on | experimental: depression instead of silencing |

Why each `game` choice was made:

* **fatigue 0.05 mV** and **ALLN silenced**: the two documented fixes for the runaway loops of the
  pure model (section 3). ALLN silencing is the one that works; fatigue alone does not.
* **DAN silenced**: dopamine acts through slow metabotropic receptors in real flies; in this model
  it is a fast excitatory transmitter, so a punishment neuron would directly *excite* the MBON it
  is supposed to teach (bitter → PPL101 → MBON11 at 120 Hz through 2,311 synapses in the probe,
  section 4.5). Silencing the DANs keeps their spikes (they gate plasticity) and mutes their
  synapses.
* **kenyon_gain 1.0**: with the antennal lobe calm, KCs are dead at 0.25 (0 % active for a single
  glomerulus, 0.3 % for a four-glomerulus mix) and sparse and odour-specific at 0.75-1.0
  (section 4.3).
* **plasticity on** with a dopamine floor, because odour alone drives one PPL1 neuron (PPL102) at
  ~100 Hz in this model and must not teach the fly to avoid everything it smells (section 4.7).

The game also keeps a **watchdog** (hand-built, `game.py`): if the whole brain fires above
150,000 spikes/s for 1.5 s with no sensory input for 0.5 s, or for 4 s regardless, the brain is
reset to rest and the event is logged as "runaway firing".

---

## 2. Validation: the six classic experiments

The six experiments of Shiu et al. 2024 and the fly-brain-minecraft benches, as coded in
`experiments.py`. A readout passes when `lo ≤ rate ≤ hi + max(1, 0.15·hi)`.

| experiment | stimulus (population: Hz) | readout | expected |
|---|---|---|---|
| Silence | none, 300 ms | whole brain | 0 Hz |
| Sugar on the mouthparts | `LB3b,LB3c`: 120; `PhG1a,PhG1b,PhG1c`: 100; `LgLG3`: 80 | GNG232 (G2N-1) / DNg67 (Fudog) / MN9 / Kenyon cells | 20-60 / 10-40 / 30-90 / 0-1 Hz |
| Bitter taste | `LB1a,LB1b,LB1c,LB1d`: 120 | GNG087 (Scapula) / MN9 | 100-400 / 0-5 Hz |
| Sugar + bitter | both | MN9 | 0-5 Hz |
| Something looming on the right | `LC4/R,LPLC2/R`: 150, 400 ms | DNp01 (giant fibre) / TTMn | 250-400 / 40-100 Hz |
| Dust on the antennae | `subclass:wind_gravity,subclass:grooming`: 150 | DNg62 (aDN1) / DNge078 (aDN2) | 100-260 / 80-200 Hz |

### 2.1 Results per profile (seed 0, this machine)

Rates in Hz; "after" is whole-brain spikes/s one second after the stimulus ends
(calm < 1,000; small loop < 50,000; RUNAWAY above).

| readout | expected | pure (starter README) | pure (`fly_brain.py --profile pure`, this kit) | fatigue 0.05 only (`synaptic-depression/log_baseline`) | **game** (`--profile game`) | brakes (`--profile brakes`) |
|---|---|---|---|---|---|---|
| GNG232 | 20-60 | 37 | 37 | 40 | **40** | 24 |
| DNg67 | 10-40 | 22 | 22 | 23 | **23** | 0 * |
| MN9 (sugar) | 30-90 | 69 | 69 | 50 | **47** | 10 * |
| Kenyon cells (sugar) | 0-1 | 0 | 0 | 0 | **0** | 0 |
| GNG087 (bitter) | 100-400 | 295 | 287 | 224 | **225** | 124 |
| MN9 (bitter) | 0-5 | 0 | 0 | 0 | **0** | 0 |
| MN9 (sugar + bitter) | 0-5 | 0 | 0 | 0 | **0** | 0 |
| DNp01 (loom) | 250-400 | 343 | 343 | 295 | **293** | 195 * |
| TTMn (loom) | 40-100 | 67 | 68 | 60 | **58** | 7 * |
| DNg62 (dust) | 100-260 | 158 | 192 | 139 | **144** | 32 * |
| DNge078 (dust) | 80-200 | 118 | 139 | 104 | **106** | 19 * |
| readouts in range | | 12/12 | 12/12 | 12/12 | **12/12** | 6/12 |
| after sugar | | 0 | 0 | 350,270 | **0** | 0 |
| after bitter | | RUNAWAY | 568,850 | 262,064 | **0** | 0 |
| after sugar + bitter | | RUNAWAY | 568,194 | 257,554 | **0** | 0 |
| after loom | | | 15,862 | 4,806 | **3,944** | 710 |
| after dust | | RUNAWAY | 136,988 | 0 | **6,740** | 0 |

Notes:

* The two pure columns differ slightly (bitter 295 vs 287 Hz, dust 158/118 vs 192/139 Hz, dust
  after-activity 546,366 vs 136,988 spikes/s in `log_baseline` vs this run) because this kit's
  simulator draws its random numbers in a different order from the starter's; same model, different
  Poisson realisation. The pure sugar experiment is itself seed-sensitive: in one of three repeats
  in `synaptic-depression/log_perf` a different draw ignited the antennal-lobe loop (147,000
  spikes in 600 ms instead of 31,000-34,000).
* The `game` profile passes all twelve readouts and every after-stimulus period is calm or a
  small loop. The small loops after loom and dust are the `DNg33`/serotonin loop the starter
  README describes; they do not move the body.
* The `brakes` profile fails half the readouts (starred). Depression is applied to every neuron,
  including the Poisson-clamped sensory afferents, so a 120 Hz taste neuron's synapses run down to
  a fraction of their strength within 100 ms (section 3.3). It is kept so the trade-off can be seen.
* Fatigue alone (column 5) keeps the readouts but not the loops: even sugar leaves a 350,000
  spikes/s runaway. That is why ALLN silencing was needed.

### 2.2 Other pathways checked in the starter kit

These came out of scanning ~400 sensory types with the pure model (starter README, section 4)
and are used by the game:

* `LC10a` on one side → `DNa02` on the same side (via `AOTU019`, `AOTU041`, `TuTuA`): the lure.
* head bristles `BM_InOm` → `MDN` (backing away from a bump). The bristles adapt (v2.4): a touch
  fires them for 0.2 s, and a head held still against the wall fires them again only when it moves
  or after 2.5 s. Before that a sustained wall touch drove the grooming neurons, grooming stopped
  the fly with its head on the wall, and it groomed there indefinitely.
* `pC1` courtship neurons → `pIP10` (song).

---

### 2.3 Five runs, and no learning carried between them (v2.7)

Two things made single verdicts less trustworthy than they looked.

* **Learning leaked from one run into the next.** `reset()` clears the learning traces but not
  what was learned, and a survival run puts every experiment through one brain in turn. With the
  parts list on, the dopamine neurons fire during an odour, so a run could depress the Kenyon-cell
  synapses of the next. Running one brain on five seeds in a row, MBON11 read 58, 53, 43, 10 and
  0 Hz; with a fresh fly for each seed it was a steady 55-59 Hz. Since v2.7, `run_experiment` starts
  every seed from what the fly had learned when the experiment began and leaves the fly as it found
  it. Learning *during* a run still counts, as it does in a real fly.
* **One seed is one draw.** Most readouts are small populations measured over half a second: two
  MBON11 cells over 500 ms can only read 0, 1, 2 ... Hz, so a range edge at 1 Hz is a coin flip.
  The verdict on a fly (`survival()`, the Genome card's re-test, `fly-brain`, `--genome-sweep`) now
  uses five seeds by default. An experiment passes when every readout's mean is in range, as
  before. It is marked **fragile** when it passes on the mean but some seed on its own misses. The
  Genome card shows a fragile reflex with an amber square and names the readout; `fly-brain`'s
  summary lists them. Five seeds cost about 3 s per experiment with the compiled integrator.

The ranges are unchanged. The model's absolute rates are not comparable with recordings
(section 10), so there is no measured value to move most of them to; the fragile mark says where a
range edge sits inside the model's own spread instead. The tables elsewhere in this document were
measured on seed 0 and are left as they were measured.

The validated experiments on five seeds (game profile, v2.7):

| configuration | passed | fragile (per-seed values of the readout that misses) |
|---|---|---|
| parts list off | 16 / 16 | vinegar: MBON11 1, 0, 3, 3, 1 Hz (range 1-60) |
| parts list on (default) | 16 / 16 | courtship command: DNp13 8.8, 2.5, 11.2, 10.0, 6.2 Hz (range 5-90); fruitless neurons silenced: ps1 song motor neurons 0, 10, 2.5, 5.0, 0 Hz (range 0-3) |
| parts list, APL global (`--global-apl`) | 16 / 16 | the two above, and vinegar: MBON11 13, 1, 0, 0, 0 Hz |

With the parts list on, the vinegar readout that section 8.2 fixed holds on every seed. Two readouts
that single runs had passed are fragile. DNp13 loses the fast octopaminergic and serotonergic drive
it had in the standard model (section 8), so the courtship command sits close to its floor. More
worrying is the leak past the silenced fruitless neurons: on two seeds of five, the song motor
neurons fire 5-10 Hz although the neurons that drive them are silenced. These are the next things to
look at.

## 3. Runaway loops and brakes

### 3.1 The problem

In the pure model bitter taste, dust and every odour set off a seizure in the antennal lobe (AL)
that never stops. Measured in `olfaction-learning/q1` (pure, ORN_DM1 at 100 Hz): the whole brain
at 559,000 spikes/s during the odour and 548,000 spikes/s 500-1000 ms after it stops, of which the
AL alone contributes 184,000; 80 % of all projection neurons (PNs) fire at ~135 Hz *regardless of
which odour was given*. Any odour, single glomerulus or mixture, gives the same picture, so the
pure model's olfactory code is a runaway signature, not an odour response. Looming leaves a
smaller loop (15,800 spikes/s) between `DNg33` and a few serotonergic neurons.

### 3.2 What was tried

| brake | configuration | six-experiment readouts in range | loops after stimulus (sugar / bitter / sugar+bitter / loom / dust, spikes/s) | verdict |
|---|---|---|---|---|
| none (pure) | | 12/12 | 0 / 572k / 566k / 16k / 546k | the baseline |
| fatigue only | 0.05 mV/spike | 12/12 | 350k / 262k / 258k / 5k / 0 | readouts kept, loops not |
| fatigue + ALLN silenced (game) | 0.05 mV + `class:ALLN` | 12/12 | 0 / 0 / 0 / 4k / 7k | **adopted** |
| depression on every neuron | U 0.2-0.5 × tau_rec 200-1000 ms, no fatigue (9 points) | 4/12 at every point | all calm | ends every loop, but the clamped afferents deplete (LB3b's resource falls to x ≈ 0.13 within 100 ms at U 0.2/200 ms): GNG232, DNg67, MN9, TTMn, DNg62, DNge078 all 0 Hz |
| depression on every neuron + fatigue | same grid + 0.05 mV | 4/12 | all calm | fatigue adds nothing |
| depression, clamped afferents exempt | U 0.2-0.5 × tau 200-1000, U = 0 for the 14 stimulated populations | 6/12 | all calm | second-order interneurons that fire at 40-300 Hz depress within the 100 ms warm-up: MN9 0-1, TTMn 0, DNg62 0, DNge078 0; GNG232 rises to 79-93 (its inhibitory input depresses too) |
| mild depression, exempt | U 0.05 / tau 100 | 11/12 (TTMn 32) | 0 / 169k / 168k / 2k / 167k | best readouts, loops only reduced to ~30 % |
| mild depression, exempt | U 0.1 / tau 100 | 10/12 (TTMn 13, DNge078 68) | 0 / 0 / 0 / 1k / 101k | bitter loops end without ALLN silencing; dust loop survives |
| mild depression, exempt | U 0.1 / tau 200 (seed 0; seed 1 gives 8/12) | 7/12 (MN9 15, TTMn 0, DNg62 77, DNge078 40) | all calm | the mildest setting that ends every loop without silencing |
| mild depression, exempt | U 0.1 / tau 500 | 6/12 (MN9 1, DNg62 4, DNge078 1) | all calm | too strong |
| mild depression, exempt + fatigue | U 0.05-0.1 × tau 100-500 + 0.05 mV | 7-9/12 | partly calm | fatigue lowers every excitatory readout by 10-25 % and never raises the count |
| mild depression, every neuron | U 0.05 / tau 100 | 10/12 (DNg67 6, TTMn 28) | 0 / 169k / 169k / 2k / 0 | without exemption even U 0.05 halves the clamped drive (x_ss = 0.62 at 120 Hz) |
| mild depression, every neuron + fatigue | U 0.1 / tau 100 + 0.05 mV | 6/12 | 0 / 56k / 0 / 0 / 0 | closest grid point to the kit's `brakes` profile |
| depression exempt + ALLN silenced | U 0.1 / tau 100, fatigue 0 | 10/12 (TTMn 13, DNge078 75) | 0 / 0 / 0 / 1k / 0 | the probe's recommendation if depression is wanted at all |

All rows from `synaptic-depression` (`summarize.py` over `results_*.jsonl`; seed 0; the
"exempt" variant sets U = 0 for `LB3b,LB3c,PhG1a,PhG1b,PhG1c,LgLG3,LB1a,LB1b,LB1c,LB1d,LC4/R,
LPLC2/R,subclass:wind_gravity,subclass:grooming`). No point of the requested grid keeps all
twelve readouts in range: the expected ranges were calibrated on a non-depressing model, and the
strength of depression that kills a loop (steady-state resource `x_ss ≈ 1/(1 + U·f·tau_rec)`)
also weakens the two- and three-synapse reflex chains whose interneurons fire at 40-300 Hz. A
compensating global gain increase (0.65 → 0.75) was not tested.

### 3.3 Why depression is only the experimental profile

The kit's `FlyBrain` implements Tsodyks-Markram depression per presynaptic neuron, lazily (only
spiking neurons are touched, so the cost is O(spikes + edges), +15-25 % wall time while
stimulated and 3-7x *less* afterwards because there is no seizure to simulate). But it has no
"exempt the clamped afferents" switch: the Poisson-driven sensory neurons are ordinary neurons
and their synapses depress like any other. The `brakes` profile (U 0.1, tau_rec 100 ms, fatigue
0.05, nothing but DANs silenced) therefore lands near the "every neuron + fatigue" row above, and
on this machine gives 6/12 (section 2.1): sugar reaches MN9 at 10 Hz instead of 47, the giant
fibre at 195 Hz instead of 293, grooming at 32/19 Hz instead of 144/106. Every loop ends. That is
the honest trade-off, and it is why the game keeps silencing.

Depression in the pure model *does* end the bitter and dust runaways without touching the ALLNs
(every U ≥ 0.1, tau_rec ≥ 200 ms point); it is the readouts, not the loops, that constrain the
choice.

### 3.4 Background noise

Real brains are never silent; this model is. Two kinds of noise were measured
(`synaptic-depression/run_noise.py`, 2 s with no stimulus, seed 0):

| noise | ALLN | result |
|---|---|---|
| Gaussian membrane noise, sigma 0.2-0.44 mV/√ms | not silenced | 0-48 spikes/s total: a handful of optic-lobe cells and 4 KCs, no descending neuron |
| sigma 0.46 | not silenced | 110-142 spikes/s (71 neurons, ALLN `lLN2P` at 0.1 Hz), no descending neuron: the largest stable level |
| sigma 0.48 | not silenced | 264 → 1,442 spikes/s and growing; 31 descending neurons flicker at 2-6 Hz |
| sigma 0.5 | not silenced | 33k → 65k spikes/s plateau: an antennal-lobe hum (ALLN 79 Hz mean, DNb05 at 150 Hz) |
| sigma 0.5 | silenced | ~700 spikes/s steady (349 neurons; DNge033 and DNp102 at 2 Hz) |
| sigma 0.6 | silenced | 18k → 41k spikes/s and growing: the central complex (PEN, EPG, Delta7, EL) seizes |
| Poisson kicks, 1 mV at 1 or 3 Hz per neuron | | 0 spikes: 1 mV kicks at a few Hz never sum to 7 mV within tau_m = 20 ms |

With noise on, the six experiments cost 3x the wall time (176,422 Gaussian draws per 0.5 ms step
and no quiet-skip), the post-bitter loop is kept alive at 36-60k spikes/s unless the ALLNs are
silenced, and the "a few descending neurons flicker, nothing seizes" window is a few hundredths
of a mV wide. The kit implements the cheap Poisson-kick variant (`noise_hz`, `noise_mv`), off in
every profile; `--noise HZ:MV` switches it on for anyone who wants to look.

### 3.5 What still rings

With the ALLNs silenced, an odour no longer runs away in the antennal lobe (0-6 spikes/s there
after the odour), but a strong odour mixture kicks the **central-complex heading circuit** into a
self-sustained state: 16,000-23,000 spikes/s brain-wide right after the odour (`PEN_a`, `PEN_b`,
`EPG`, `Delta7`, `EL`, `GLNO`, `AN09A005`), decaying to 5,000 after 0.5 s and ~1,400 after
1-1.5 s (`olfaction-learning/q1c`, probe-game, kenyon_gain 0.5). It is not an AL loop, it
decays, and it is the next circuit to seize under noise (table above); it may bias heading
readouts for a second after a strong smell.

---

## 4. Olfaction and learning

### 4.1 From an odour to receptor neurons (hand-built)

Each odour in the game activates a hand-chosen set of glomeruli (`senses/olfaction.py`; the
combinations are coarse literature choices, four to five glomeruli each, so that the Kenyon-cell
code is sparse but not empty). The ORN types are the `ORN_<glomerulus>` cell types of neuPrint.

| odour | glomeruli (receptor) | label | source of the choice |
|---|---|---|---|
| apple cider vinegar | DM1 (Or42b), DM4 (Or59b), VM7d (Or42a), DP1m (Ir64a), VA2 (Or92a) | attractive | Hallem & Carlson 2006 / DoOR |
| banana / isoamyl acetate | DM2 (Or22a), DM3 (Or47a), VM2 (Or43b), DC2 (Or13a), VA6 (Or82a) | attractive | Hallem & Carlson 2006 / DoOR |
| geosmin + CO2 | DA2 (Or56a), V (Gr21a/Gr63a), DL4 (Or49a/Or85f), DL5 (Or7a), DC4 (Ir64a) | aversive | Stensmyr et al. 2012 (geosmin); Suh et al. 2004 (CO2) |
| yeast / fermentation | DM5 (Or85a), VC1 (Or33c/Or85e), VA1v (Or47b), DL1 (Or10a), VM5d (Or85b) | attractive | Hallem & Carlson 2006 / DoOR |

The rate is `90 Hz · c^1.5 / (0.25^1.5 + c^1.5)` of the adapted concentration (a Hill saturation;
real ORNs reach ~200 Hz, the model's antennal lobe was calibrated for ~80 Hz). ORNs adapt to a
steady background with a 3 s time constant (Nagel & Wilson 2011). The plume itself is a filament
model of turbulent odour (puffs released at 6 + 0.2·wind puffs/s, drifting with the wind and
spreading), so what the fly smells flickers. The "innate" label is only shown on screen and used
by the hand-built steering (section 4.9); it never touches the brain.

Only the vinegar set's core four glomeruli (`ORN_DM1,ORN_DM4,ORN_VM7d,ORN_DP1m`, the probe's
"MIX") were measured in the probe; the other three odours were not, and KC recruitment depends
strongly on which glomeruli are driven (section 4.3).

### 4.2 The antennal lobe: pure versus probe-game

`olfaction-learning/q1`, kenyon_gain 0.25, seed 0:

| stimulus | config | own PN(s) | other PNs | KCs active | after odour (brain / AL spikes/s) |
|---|---|---|---|---|---|
| ORN_DM1 at 100 Hz | pure | DM1_lPN 483 Hz | mean 135 Hz, 80 % active | 5.7 % | 548,000 / 184,000 |
| ORN_DM1 at 100 Hz | probe-game | DM1_lPN 339 Hz | mean 3.0 Hz, 5 % active | 0 % | 0 / 0 |
| MIX at 80 Hz | pure | DM1_lPN 467, DM4_adPN 369, DP1m_adPN 492, VM7d_adPN 234 | 137 Hz, 81 % | 5.9 % | 546,000 / 184,000 |
| MIX at 80 Hz | probe-game | DM1_lPN 301, DM4_adPN 170, DP1m_adPN 386, VM7d_adPN 164 | 8.8 Hz, 13 % | 0.3 % | 23,000 / 6 |
| ORN_DL5 at 100 Hz | probe-game | DL5_adPN 275 | 0.9 Hz, 2 % | 0 % | 0 / 0 |

With the local neurons silenced the antennal lobe is odour-specific and calm again 500 ms after
the odour. But the Kenyon cells are dead at the default input gain.

### 4.3 Kenyon-cell sparseness versus `kenyon_gain`

Probe-game, 600 ms odour, "active" = at least one spike in the 500 ms window, Jaccard = overlap of
active-KC sets (`olfaction-learning/q1`, `q1_game_hi`, `q6_extras`):

| kenyon_gain | DM1 (100 Hz) | MIX (80 Hz) | DL5 (100 Hz) | Jaccard DM1-MIX (MIX contains DM1) | Jaccard DM1-DL5 / MIX-DL5 |
|---|---|---|---|---|---|
| 0.25 | 0 % | 0.3 % (13 KCs) | 0 % | | |
| 0.5 | 1.2 % (49) | 4.8 % (194, max 46 Hz) | 0.02 % (1) | 0.105 | 0.000 / 0.000 |
| 0.75 | 3.0 % (120, max 54 Hz) | 7.9 % (323, mean 1.64 Hz, max 64 Hz) | 0.3 % (13) | 0.188 | 0.000 / 0.000 |
| **1.0** | 4.3 % (176) | 8.6 % (351, mean 2.19 Hz, max 78 Hz) | 0.6 % (26) | 0.223 | 0.005 / 0.000 |
| pure, 0.25 (for comparison) | 5.7 % (231) | 5.9 % (241) | | 0.903 | 0.799 / 0.766 |

Real flies activate roughly 5-10 % of their KCs per odour, so 0.75-1.0 is in range; the game uses
1.0. Reliability of the code across seeds at 0.75: Jaccard 0.952 (DM1) and 0.935 (MIX) between
seed 0 and seed 1. A second four-glomerulus mixture with no glomerulus in common with MIX
(`ORN_DL5,ORN_DC1,ORN_VA2,ORN_DM2`, "MIX2") recruits only 2.1 % of KCs (85 cells) and overlaps
MIX with Jaccard 0.02; MIX2 across seeds 0.89. Single glomeruli are weak: DL5 at 200 Hz gives
0.6 %, DA1 at 100 Hz 0.9 %, DM1 at 200 Hz 5.3 %.

The pure model's code is "sparse" only in count: the same KCs fire for every odour (Jaccard
0.77-0.90 between unrelated odours) because the antennal lobe saturates.

KC subtypes are recruited very unevenly (probe-game, kenyon_gain 0.75, MIX): KCab-m 22 %,
KCa'b'-m 19 %, KCab-c 12 %, KCab-s 8 %, KCg-m 4 %, KCa'b'-ap2 2 %; KCg-d, KCab-p and KCa'b'-ap1
never fire. This is unlike real flies, where the γ lobe (KCg-m, 1,342 cells here) is the main
site of short-term memory, and it biases which compartments can learn (section 8).

### 4.4 Which dopamine neurons teach which output neuron: read from the wiring

`olfaction-learning/q3_wiring.txt` (a query, no simulation): for every MBON type, its
neurotransmitter, its KC input, and its three strongest dopamine (DAN) inputs by direct synapses.
The PPL1 cluster is the punishment cluster and the PAM cluster the reward cluster (Aso et al.
2014a). Every MBON type with dopaminergic input has one dominant cluster, so the compartment map
falls straight out of the wiring; `plasticity.py` uses exactly these DAN→MBON synapses (weighted)
as each MBON's dopamine source.

| MBON | n | transmitter | KC→MBON synapses (KCs) | strongest DAN inputs (synapses) | PPL1 / PAM total |
|---|---|---|---|---|---|
| MBON01 | 2 | glutamate | 40,574 (1,963) | PAM01 1569, PAM02 546, PAM15 70 | 0 / 2222 |
| MBON02 | 2 | glutamate | 12,255 (1,251) | PAM04 1152, PAM02 22, PAM03 14 | 0 / 1203 |
| MBON03 | 2 | glutamate | 23,183 (694) | PAM06 3292, PAM05 955, PAM02 210 | 0 / 4633 |
| MBON04 | 2 | glutamate | 4,526 (561) | PAM05 717, PAM06 661, PPL107 265 | 265 / 1628 |
| MBON05 | 2 | glutamate | 35,620 (1,569) | PAM08 2176, PAM07 860, PAM12 155 | 36 / 3265 |
| MBON06 | 2 | glutamate | 24,929 (1,675) | PAM10 3274, PAM09 381, PAM11 180 | 24 / 3851 |
| MBON07 | 4 | glutamate | 19,484 (1,531) | PAM11 1918, PAM10 35, PAM09 10 | 0 / 1963 |
| MBON09 | 4 | GABA | 63,018 (2,248) | PAM12 1802, PAM13 862, PAM14 785 | 11 / 3514 |
| MBON10 | 9 | GABA | 566 (89) | PAM13 78, PAM14 21, PPL107 11 | 11 / 99 |
| MBON11 | 2 | GABA | 38,852 (2,854) | PPL101 2311, PPL102 128, PAM10 70 | 2445 / 100 |
| MBON12 | 4 | acetylcholine | 17,698 (1,541) | PPL103 802, PPL105 7, PPL107 5 | 814 / 0 |
| MBON13 | 2 | acetylcholine | 9,972 (593) | PPL105 657 | 657 / 0 |
| MBON14 | 4 | acetylcholine | 19,079 (1,313) | PPL106 860 | 860 / 0 |
| MBON15 | 4 | acetylcholine | 154 (27) | PPL103 117, PPL107 6 | 123 / 0 |
| MBON15-like | 4 | acetylcholine | 1,676 (231) | PPL103 140, PPL105 76 | 216 / 0 |
| MBON16 | 2 | acetylcholine | 3,016 (368) | PPL104 302 | 302 / 0 |
| MBON17 | 2 | acetylcholine | 2,443 (330) | PPL104 118 | 118 / 0 |
| MBON17-like | 2 | acetylcholine | 1,134 (174) | PPL104 49, PPL105 6 | 55 / 0 |
| MBON18 | 2 | acetylcholine | 13,040 (1,368) | PPL105 434, PPL201 5 | 434 / 0 |
| MBON19 | 4 | acetylcholine | 790 (106) | PPL105 98 | 98 / 0 |
| MBON20 | 2 | GABA | 2,169 (332) | PPL101 93, PPL102 38, PPL103 7 | 138 / 0 |
| MBON21 | 2 | acetylcholine | 6,489 (899) | PAM08 523, PAM07 228, PAM01 93 | 10 / 868 |
| MBON22 | 2 | acetylcholine | 12,172 (1,632) | PPL202 30 | 0 / 0 |
| MBON23 | 2 | acetylcholine | 764 (133) | PPL105 125 | 125 / 0 |
| MBON24 | 2 | acetylcholine | 728 (129) | PAM04 469, PAM01 6 | 0 / 475 |
| MBON25 | 2 | glutamate | 62 (12) | PPL101 70, PPL102 34, PPL103 17 | 121 / 0 |
| MBON25-like | 4 | glutamate | 35 (7) | PPL101 39, PPL103 28, PPL102 14 | 81 / 0 |
| MBON26 | 2 | acetylcholine | 4,001 (216) | PAM05 1123, PAM08 114, PPL107 40 | 40 / 1264 |
| MBON27 | 2 | acetylcholine | 5,067 (224) | PAM01 255, PAM08 139, PAM07 47 | 12 / 441 |
| MBON28 | 2 | acetylcholine | 1,822 (209) | PPL104 167 | 167 / 0 |
| MBON29 | 2 | acetylcholine | 1,480 (242) | PAM08 254, PAM01 69 | 0 / 323 |
| MBON30 | 2 | glutamate | 10,945 (1,170) | PPL103 187, PPL101 101, PPL102 23 | 316 / 6 |
| MBON31 | 2 | GABA | 10,243 (636) | PPL103 681, PPL107 20, PPL105 17 | 723 / 6 |
| MBON32 | 2 | GABA | 11,452 (1,249) | PPL103 972, PPL102 73, PPL101 14 | 1064 / 0 |
| MBON33 | 2 | acetylcholine | 760 (95) | PPL103 263, PPL102 57, PAM12 37 | 343 / 50 |
| MBON34 | 2 | glutamate | 0 (0) | none | 0 / 0 |
| MBON35 | 2 | acetylcholine | 2,652 (347) | PPL103 676, PPL102 162, PPL108 107 | 1017 / 0 |

PPL1-paired types (21 types, 54 neurons): MBON11, 12, 13, 14, 15, 15-like, 16, 17, 17-like, 18,
19, 20, 23, 25, 25-like, 28, 30, 31, 32, 33, 35. PAM-paired (14 types): MBON01, 02, 03, 04, 05,
06, 07, 09, 10, 21, 24, 26, 27, 29. MBON22 (calyx) and MBON34 have no PAM/PPL1 input and never
learn. The DAN types with most synapses onto MBONs: PAM06 3980, PPL103 3916, PAM08 3548, PAM10
3385, PAM05 2808, PPL101 2697. The hemibrain-style compartment names (MBON14 = α3 etc.) do not
match this dataset's pairing (MBON14 is paired with PPL106, MBON12 with PPL103, MBON13/18 with
PPL105, MBON16/17/28 with PPL104); the kit uses the wiring, never the name glosses.

The DANs themselves are fed mostly by KCs: the top inputs of the PPL1 cluster are KCg-m (17,001
synapses), KCa'b'-ap1 (5,545), KCa'b'-ap2 (4,261), KCab-m (3,570); of the PAM cluster KCa'b'-ap1
(6,388), KCab-p (3,883), KCg-m (3,047), DPM (2,300).

### 4.5 What reward and punishment signals the wiring carries

`olfaction-learning/q2`, `q2b` (simulation and wiring queries):

| signal | stimulus | pure | probe-game | wiring |
|---|---|---|---|---|
| reward: sugar → PAM | the sugar experiment; also 2x rates; also all sugar GRNs at 150 Hz | PAM 0 % active, 0.0 Hz (MN9 69) | PAM 0.0 Hz (MN9 50-69) | 0 direct sugar-GRN→PAM synapses; 400 two-hop synapses onto 26 of 316 PAMs (PAM11 314, PAM01 65) via PRW072 (299) and PRW044 (80) |
| punishment: bitter → PPL1 | `LB1a-d` at 120 Hz | PPL1 mean 79 Hz (PPL102 214, PPL101 127, PPL107 115, PPL108 110, PPL106 69) but the brain is in runaway (572k spikes/s), so this is not bitter-specific | PPL1 mean 34 Hz: **PPL101 91**, PPL102 88, PPL107 50, PPL108 22, PPL106 10, PPL103 9; PAM 0; brain calm after | 0 direct, 48 two-hop synapses (PPL106 30, PPL101 18) |
| odour alone → DANs | MIX at 80 Hz | PPL102 159, PPL107 108, PPL108 84 (runaway) | **PPL102 101**, PPL107 39, PPL204 20, PPL203 17, PPL104 9, PPL101 7; PAM < 0.5 | |
| odour alone → DANs | DM1 at 100 Hz | | PPL203 13, PPL107 9; nothing else | |

So in this model **punishment is in the wiring** (bitter taste drives the PPL1 cluster, and
PPL101, the γ1pedc neuron that waters MBON11, is bitter-specific: 91 Hz to bitter, 7 Hz to MIX,
0 to DM1), while **reward is not**: sugar never fires a single PAM neuron in either configuration.
The game therefore:

* injects reward by hand: while the fly eats sugar (and learning is on) `prefix:PAM` is driven at
  40 Hz (`REWARD_HZ`); labelled on screen as hand-built;
* lets punishment come from the wiring when the fly tastes bitter, and offers a "shock" tool that
  drives `prefix:PPL1` at 80 Hz for 1 s, the classic conditioning stimulus (Tully & Quinn 1985);
* silences the DAN class so that PPL101's 2,311 excitatory synapses onto MBON11 cannot drive
  MBON11 directly (in the probe, bitter alone pushed MBON11 to 120 Hz that way; only KC-driven
  activity is meaningful for a readout);
* keeps a dopamine floor (section 4.7) because PPL102 fires at ~100 Hz to an odour by itself.

### 4.6 The learning effect, measured

`olfaction-learning/q5`: the KC→MBON synapses onto the PPL1-paired MBONs were scaled by 0.5 (or
0) in a copy of the model, either for all KCs or only for the 323 KCs active during MIX, and the
MBON rates to MIX and to the unpaired odour DM1 were re-measured. Probe-game, kenyon_gain 0.75
and 1.0, seed 0 (seed-to-seed noise floor of the baseline in brackets):

| MBON (compartment cluster) | gain 0.75: MIX before → ×0.5 → ×0 | gain 1.0: MIX before → ×0.5 → ×0 | DM1 (unpaired) before → ×0.5 | noise floor (seed 0 / seed 1) |
|---|---|---|---|---|
| MBON11 (PPL1, GABA) | 31 → 14 → 7 | 22 → 7 → 3 | 0 → 0 | 31/27, 22/33 |
| MBON14 (PPL1, ACh) | 26 → 3 → 0 | 40 → 12 → 0 | 4 → 0 (gain 1.0) | 26/24, 40/42 |
| MBON13 (PPL1, ACh) | 2 → 0 → 0 | 2 → 0 → 0 | | |
| MBON31 (PPL1, GABA) | 51 → 43 → 41 | 47 → 46 → 39 | 15 → 14 | 51/51, 47/50 |
| MBON28 (PPL1, ACh) | 44 → 45 → 43 | 41 → 42 → 39 | 4 → 4 | 44/40 |
| MBON30 (PPL1, glutamate) | 79 → 93 → 100 | 96 → 94 → 94 | | 79/0 (bistable) |
| MBON20 (PPL1, GABA) | 30 → 27 → 25 | 27 → 27 → 25 | | 30/34 |

Scaling only the MIX-active KCs' synapses gives the same numbers as scaling all of them, as it
must (inactive KCs contribute nothing). Scaling the PAM-side synapses instead (reward learning)
drops MBON09 from 18 to 6 Hz, MBON02 from 4 to 0, MBON07 from 6 to 0 (gain 0.75), and at gain 1.0
MBON02 14 → 0, MBON07 13 → 0, MBON09 14 → 0. In the pure model the same manipulation changes
nothing visible (MBON30 200 → 187, MBON28 159 → 157, MBON20 50 → 59), because those rates are
runaway-driven.

Two lessons shape the game's readouts: depression is clearly visible only in MBONs whose input is
dominated by KCs, and MBON30/28/31/20 respond to odour even when the KCs are silent
(`q3b_mbon_inputs`: MBON30 63 % KC input, the rest from MBON05, CRE067, FR1; MBON28 38 %, with
direct PN input; MBON31 41 %, mostly lateral horn; MBON20 26 %; MBON26 20 %; MBON10 15 %; MBON11
and MBON09 89 %). The on-screen "approach MBONs" bar is `MBON11,MBON12,MBON14,MBON09` and the
"avoidance MBONs" bar `MBON01,MBON02,MBON05,MBON06,MBON07`, all KC-dominated.

### 4.7 The learning rule as implemented (`plasticity.py`)

Everything structural is read from the wiring: the plastic synapses are every KC→MBON connection
(one scale factor per connection), and the dopamine reaching an MBON is the synapse-weighted mean
rate of the DANs that synapse directly onto it (table 4.4). Only the rule's constants are
hand-chosen:

| constant | value | meaning | basis |
|---|---|---|---|
| `kc_tau_ms` | 1500 ms | how long a KC spike leaves its synapses eligible (eligibility trace; ~3 recent spikes = full) | forward pairing window of 1-2 s (Hige et al. 2015; Handler et al. 2019) |
| `dan_tau_ms` | 400 ms | how long a DAN spike's dopamine lingers | hand-chosen |
| `dan_scale` | 40 spikes/s | DAN rate in a compartment that counts as "full" dopamine | hand-chosen |
| `dan_floor` | 0.5 | dopamine below half of full does nothing | PPL102 fires ~100 Hz to odour alone (section 4.5); with the synapse weighting, odour alone leaves MBON11's dopamine below the floor while bitter or shock saturate it |
| `rate` | 0.5 /s | fraction of remaining strength lost per second at full eligibility × full dopamine | a strongly paired odour loses half its drive in ~1 s |
| `floor` | 0.15 | a synapse never falls below 15 % of its original strength | hand-chosen |
| `recover_min` | 20 min | depressed synapses recover most of their strength over this time (short-term memory) | hand-chosen; fly STM fades over tens of minutes |
| `block_steps` | 20 (10 ms) | update granularity | performance |

Per 10 ms block, for every plastic synapse: `scale *= 1 − rate · 0.01 · elig(KC) · dopamine(MBON)`,
clipped at `floor`, then `scale += (1 − scale) · 0.01 / (20·60)` for recovery. This is
dopamine-gated depression of the *active* KC's synapses only (heterosynaptic plasticity, Hige et
al. 2015); backward pairing (dopamine before odour), which potentiates in real flies (Handler et
al. 2019), is not implemented. Memories survive a "new fly" reset and can be erased with
"forget"; weights can be saved and loaded. The learning panel reports, per MBON type, the mean
remaining strength of all its KC synapses and, weighted by the Kenyon cells active right now,
the strength the current odour actually experiences.

### 4.8 Valence readout and its caveats

`plasticity.py` assigns each MBON a valence by transmitter, following the rule of Aso et al.
2014b: glutamatergic MBONs promote avoidance, GABAergic and cholinergic ones approach. With the
compartment table this gives the textbook picture: punishment (PPL1) depresses the KC drive onto
approach MBONs, so the odour is avoided; reward (PAM) depresses the drive onto avoidance MBONs,
so the odour is approached. The transmitter rule is a first approximation and has exceptions in
the literature.

Whether MBON activity can *steer* the fly through the wiring was tested (`q4`, `q4b`, `q6a`):

| stimulus (80 Hz, 500 ms) | config | descending neurons |
|---|---|---|
| all 26 glutamatergic ("avoidance") MBONs | pure and probe-game | 0 of 1,314 DNs fire; the whole brain's 2,040 spikes/s are the stimulated MBONs themselves |
| all 71 cholinergic + GABAergic ("approach") MBONs | probe-game | DNg33 164, DNg66 54, DNge150 30, DNg16 26, DNg13 24, DNge138 24 ...; DNa02 0, DNp09 0, MDN 0 (94 DNs active) |
| cholinergic MBONs only | probe-game | DNg33 162, DNg49 36, DNg66 32 ...; DNa02/L 2 |
| GABAergic MBONs only | probe-game | nothing (1,622 spikes/s) |
| cholinergic + glutamatergic together | probe-game, seeds 0-2 | DNa02/R 22-26 Hz, DNa03 10-15, MDN 0-5: the glutamatergic MBONs act only by disinhibition on top of other activity |
| approach MBONs | pure | runaway (559,000 spikes/s) |

An inhibitory population cannot drive anything from rest, and the approach group reaches
`DNg33`, `DNg66` and `DNg13` but not the steering neuron `DNa02` or the walking neuron `DNp09`.
The valence readout in the game is therefore **hand-built**: `Game.odour_steering()` reads the
learned bias directly off the plastic synapses of the KCs active now (eligibility-weighted mean
depression of avoidance-MBON synapses minus that of approach-MBON synapses, times 1.2), adds the
literature valence label (±0.35), and turns the fly up the concentration gradient between its two
antennae, or upwind when there is no gradient. What the brain contributes is which KCs are active
and how depressed their synapses are; the turn itself is scripted.

### 4.9 Conditioning protocols

`scenarios.py` scripts the two classic assays as timed protocols: appetitive conditioning (two
odours, one paired with sugar while hungry, then a test without food; Tempel et al. 1983) and
aversive conditioning (one odour paired with the shock tool, then a test; Tully & Quinn 1985),
each with a baseline period and a time-weighted preference index between the two sources. The
protocol only places things and injects the labelled signals; whether the preference changes is up
to the KC code, the plasticity rule and the hand-built steering above.

---

## 5. Computed vision

The starter kit *injected* vision (the looming detectors were driven from the pointer's angular
size). This kit has a retina, and the probe `computed-vision` measured how far real columnar
motion detection can reach through the wiring.

### 5.1 The retina (hand-built)

Each eye is a grid of 30 azimuth × 16 elevation facets (480 per eye) looking from 8° across the
midline to 165° to the side and from −55° to +60° elevation, with alternate rows offset by half a
facet (~6° per facet). Every tick each facet reports the brightness in its direction: floor 1.0,
wall 0.55 (a 30 mm high grey wall, or a striped drum of 0.85/0.25 bands when the optomotor
stimulus is on), posts 0.1, food 0.5, the female 0.12, the lure 0.12, the hand 0.1, all as dark
shapes of given size and height seen from an eye 1 mm above the floor. Photoreceptors are not
simulated: light on R1-R6 does nothing in this model (they inhibit L1-L3 through histamine and
those cells have no resting activity), so the image is turned into neuron rates at two later
stages.

### 5.2 Feature detectors (hand-built, in parallel with the columns)

From the two images `FeatureDetectors` computes the rates of the real visual projection neurons
that the wiring cannot be made to compute (section 5.5-5.6):

* **Looming** → `LC4`, `LPLC2` on the eye that sees it. Dark blobs (objects only, never the wall
  or its stripes) are matched frame to frame; a blob of at least 6 facets that has grown for two
  frames in a row, in place rather than shifting sideways, has an edge speed
  `e = Δradius/dt` in °/s. Above 60 °/s, `LC4 = 160 · sat((e − 60) / 300)` Hz and
  `LPLC2 = LC4 · min(1, area/14)`. An object that merely sweeps across the eye, or the fly's own
  turn, does not loom (Klapoetke et al. 2017: real looming detectors respond to expansion, not
  translation).
* **Small moving object** → `LC10a` (Ribeiro et al. 2018) and `LC11` (Keleş & Frye 2017). A blob
  of 1-14 facets whose centroid moves faster than the fly's own image shift gives
  `LC10a = 70 · (0.35 + 0.65 · min(1, deg/s / 60)) · min(1, size/2)` Hz and `LC11 = 0.6 · LC10a`.
* **Wide-field motion**: a 1-D correlation of the row-averaged image gives a horizontal flow that
  drives `T4a/T5a` (front-to-back) or `T4b/T5b` (back-to-front) uniformly at up to 40 Hz. This
  path is only used with `--no-columnar`; the game normally lets the columns do it.

`sat(x) = x^1.5 / (0.2^1.5 + x^1.5)` normalised to 1, the saturating curve of the
fly-brain-minecraft encoders. The constants are hand-chosen and the escape scenario in
`scenarios.py` (three swoops of the hand, one slow) is the regression test.

### 5.3 The column lattice and the preferred-direction axes (from the data)

The briefing assumed T4/T5 carry column coordinates; they do not (0 of 13,585). `step1b`
recovered a **home column** for 13,582 of them as the synapse-weighted centroid of the `hex1`,
`hex2` coordinates of their tagged inputs: Mi1 (65 synapses per cell), Mi9 (21), Mi4 (10), C3 (2)
for T4; Tm2 (34), Tm9 (32), Tm1 (21), Tm4 (16) for T5. Median RMS spread of a cell's inputs is
0.87 columns; the rounded centroid equals the single strongest input column in 76.5 % of cells;
about 15 % of rounded columns hold 2-3 cells of one subtype (roughly one cell per column plus
rounding). `ColumnarMotion._home_column` in `senses/vision.py` does exactly this at start-up.

The lattice: Mi1 has exactly one cell per column, 887 columns in the right eye and 875 in the
left; `hex1` runs 1-36 and `hex2` 1-39; L1 soma spacing shows (1,1) is the short diagonal
(1.22 µm) and (1,−1) the long one (1.57 µm), i.e. the hex axes are 120° apart and the eye spans
about 32 × 30 columns. Soma anatomy fixes the orientation: `hex1 + hex2` increases dorsally
(correlation with soma depth −0.97 on the left, −0.96 on the right; the calyx is dorsal at
y = 11k, the subesophageal zone ventral at y = 38-44k), and `hex1 − hex2` increases laterally and
posteriorly in the medulla, which is the *anterior* visual field after the first chiasm.

**Preferred directions** (`step1c`): the dendrite of a T4 cell samples Mi9 on its preferred
side (where an edge moving in the preferred direction enters the receptive field) and Mi4/C3 on
its null side (Takemura et al. 2017), so an edge moving in the preferred direction runs from the
Mi9 centroid to the Mi4 centroid: the **preferred direction is Mi4 − Mi9**, the offset below with
its sign reversed. Per subtype and side (hex units; standard error ≤ 0.03; 440-764 cells each):

| subtype | Mi9 − Mi4, left | Mi9 − Mi4, right | preferred direction (Mi4 − Mi9) in the 60° hex plane (L / R) | direction |
|---|---|---|---|---|
| T4a | (+0.85, −1.17) | (+0.91, −1.16) | +105° / +108° | front-to-back |
| T4b | (−0.88, +1.28) | (−0.84, +1.34) | −78° / −82° | back-to-front |
| T4c | (−1.74, −1.35) | (−1.79, −1.42) | +26° / +26° | up |
| T4d | (+1.71, +1.90) | (+1.74, +1.88) | −148° / −149° | down |

a/b are antiparallel along `hex2 − hex1`, c/d along `hex1 + hex2`, and the T5 offsets
(Tm9 − Tm1) point the same way. The two sides agree within 3°. `COLUMNAR_AXES` in
`senses/vision.py` uses a = +106°, c = +26° in the 60°-axis plane
(`cx = hex1 + 0.5·hex2, cy = 0.866·hex2`), projects every column onto these two axes and spreads
the result linearly over the eye's field (azimuth −8° to 165°, elevation −55° to 60°): azimuth
grows (toward the back) with `hex2 − hex1` and elevation grows with `hex1 + hex2`, exactly the
anatomy above. Distances along the lattice are not exact angles, but every column lands on the
right part of the visual field and every subtype points the right way, which is what the
downstream wiring needs. In this frame the a and c axes are 80° apart rather than 90°; the real
lattice is tilted relative to the equator. (The first release of this kit had transcribed the
Mi9 − Mi4 offsets themselves as the preferred directions, i.e. the same frame rotated by 180°,
which put the front of the eye at the back and the top at the bottom; the probes in 5.4-5.5
used the correct frame throughout, so their numbers were unaffected. It was caught by the three
checks below, which the test suite now encodes on the synthetic lattice.)

Three independent checks that the frame is right:

* **T4/T5 → lobula-plate tangential cells** (right side, synapses): T4a → HSE 6951, HSN 6526,
  HSS 6741 (and LPi12 28,678); T4b → H2 12,235 (HS 0); T4c → LPi34 39,231 (VS 6); T4d → VS 24,437;
  T5a → HSE 8317, HSN 8219, HSS 7037; T5b → H2 14,558; T5d → VS 24,463. So a/b are the
  horizontal system (layers 1/2) and c/d the vertical system (layers 3/4), as in Maisak et al.
  2013. And the synapse-weighted elevation of each HS cell's T4/T5 inputs in this frame is
  HSN +12°, HSE −1°, HSS −26° (right side; left +12°, +1°, −30°): north above, south below, as the
  names say (Schnell et al. 2010).
* **LPLC2 receptive fields**: for each of the 91 right LPLC2 cells the mean visual offset of its
  T4/T5 inputs from its own T4/T5 centre is a: +1.67 columns azimuth (behind), b: −1.49 (in
  front), c: +4.13 elevation (above), d: −4.41 (below); left eye +1.81 / −1.53 / +4.12 / −4.35
  (7,500-10,600 synapses per subtype). That is the outward-motion arrangement Klapoetke et al.
  2017 measured physiologically, recovered from wiring alone; and driving the outward pattern
  within 9 columns of a centre in this frame gives LPLC2 8.2 Hz (best cell 133 Hz) and DNp01
  6.7 Hz, the contraction pattern 1.0 Hz and 0 (`check_loom`, 250 ms).
* **Mi1 somata**: the medulla's dorsal edge is toward the calyx (low y in the volume). The
  frame's elevation of an Mi1 column correlates −0.97 with its soma's y on both sides, i.e. up
  is dorsal.

### 5.4 What T4/T5 injection reaches: looming

T4/T5 of the right eye were driven as static Poisson patterns for 400 ms (100 ms discarded),
150 Hz, seed 0 unless stated (`step2`, `step2b`, `step5c`, `step6`). "Outward" means: within a
radius of the loom centre, only the subtype whose preferred direction points away from the centre;
"contraction" the opposite subtype.

| stimulus (right eye) | neurons | LPLC2/R mean (max) | DNp01 pure / game | TTMn pure / game |
|---|---|---|---|---|
| reference: `LC4/R,LPLC2/R` at 150 Hz | 146 | 152 (all 91 > 50 Hz) | 345 / 298 | 70 / 60 |
| outward T4/T5, r ≤ 3 columns | 62 | 1.5 (70) | 6.7 / 1.7 | 0 / 0 |
| outward, r ≤ 6 | 261 | 5.6 (170) | 60 / 63 (seeds 1, 2: 70, 40 / 55, 62) | 3.3 / 5 |
| outward, r ≤ 9 | 583 | 11.5 (183) | 95 / 77 (seeds 1, 2: 88, 83 / 80, 77) | 10 / 5 |
| outward, r ≤ 12 | 1,027 | 12.2 (177) | 28 / 67 | 3.3 / 5 |
| outward, r ≤ 16 | 1,557 | 12.8 (173) | 33 / 75 | 1.7 / 3.3 |
| contraction, any radius 3-16 | 60-1,778 | ≤ 4.0 (max cell ≤ 43; 110-113 at r 16) | 0 / 0 | 0 / 0 |
| wide-field T4a+T5a, whole eye | 1,687 | 1.1 (50) / 2.0 (60) | 0 / 0 | 0 / 0 |
| all four subtypes, r ≤ 6 | 1,045 | 15 (12-13 cells > 50) | 60 / 65 | 1.7 / 6.7 |
| all four subtypes, whole eye | 6,795 | 44 / 51 | 87 / 155 | |
| LC4's columnar inputs (T2, TmY3, Tm4, Tm2, Tm3) r ≤ 6, alone | 643 | 0 | 57 / 78 | 18 / 17 |
| LC4 inputs r ≤ 9 + outward T4/T5 r ≤ 9 | 2,015 | 11.7 / 11.6 | 122 / 142 (seeds 1, 2: 105, 133 / 135, 147) | 32 / 28 |
| LC4 inputs, whole eye | 3,984 | 0 | 197 / 202 | 48 / 45 |

Three findings:

1. **LPLC2 → DNp01 discriminates looming through the real wiring**: outward motion drives the
   matching LPLC2 cells to 150-183 Hz and the giant fibre to 45-95 Hz; contraction and wide-field
   motion give DNp01 = 0 in every seed and at every size.
2. **It is weak**, because `LC4`, the giant fibre's largest input (3,181 synapses per DNp01
   cell, against 2,418 from LPLC2), receives essentially no T4/T5 input (3 synapses per cell). LC4
   is driven by lobula columnar cells: T2 (372 synapses per LC4 cell), TmY3 (367), Tm4 (246), Tm2
   (148), Tm3 (95). LPLC2's inputs are Tm5Y (152 per cell), T5d 79, T5b 67, T5c 63, T4c 54, T5a
   47, T4b 36, T4a 36 (396 T4/T5 synapses per cell), plus inhibitory PVLP011 and Tm37. Adding
   LC4's own inputs in the same patch gets DNp01 to 105-147 Hz and TTMn to 23-33 Hz, still short
   of the 250-400 / 40-100 Hz band.
3. A simultaneous flash of all four subtypes everywhere looks like a loom to every LPLC2 (each
   sees outward motion on all four sides): the model's surround inhibition (LPi cells) does not
   cancel it. A retina must therefore produce one subtype per column, never opposite subtypes
   together.

### 5.5 What T4/T5 injection reaches: the optomotor reflex

`step3`, all T4/T5 of one horizontal subtype pair on one eye at 100 Hz, 400 ms; rates in Hz,
pure (game in brackets):

| stimulus | HS cells (HSE/HSN/HSS) | H2 | DNa02 R / L | DNp15 R / L | other |
|---|---|---|---|---|---|
| T4a+T5a right (front-to-back on the right eye) | R: 500/500/500 (447/443/440), saturated | 0 | **187 / 0** (157 / 0) | **213 / 0** (177 / 0) | DNa06/R 137 (117), DNa01 R/L 17/47, DNbe001/R 263, DNg41/R 227, DNg33 340 both sides |
| T4a+T5a left | L: 500/500/500 | 0 | 0 / 93 (0 / 97) | 0 / 193 (0 / 157) | DNa06/L 120 (113), DNb06/L 147 |
| T4b+T5b right (back-to-front) | 0 | R: 500 | 0 / 3.3 | 0 / 187 (0 / 167) | DNg41/L 220 (183), DNb03/L 167 (158), DNb01 L/R 83/70 |
| T4b+T5b left | 0 | L: 500 | 0 / 0 | 240 / 0 (197 / 0) | DNg41/R 237 (193), DNb03/R 167 |

Seeds 1 and 2 reproduce the right-eye progressive case (DNa02/R 187, 200 pure; 157, 153 game).
Front-to-back motion on one eye excites that eye's HS cells (as in the real fly) and, through the
wiring, `DNa02` and `DNp15` on the same side: a turn that *follows* the motion, which is the
optomotor response. Regressive motion goes through `H2` to the *contralateral* `DNp15`, `DNg41`
and `DNb03` with `DNa02` silent. This is the pathway the game uses (section 5.7).

### 5.6 Small objects: a null result

`step4`, `step4b`, `step5c`:

* No lobula columnar (LC) type responds to any T4/T5 patch: LC10a, LC11, LC12, LC16, LC17, LC18,
  LC4, LC6 and every other LC = 0.0 Hz for 3-column, 7-column and 19-column patches, any subtype
  mix, both configurations. LC dendrites are in the lobula; T4/T5 only reach the lobula plate.
* A 19-column patch of one horizontal subtype (a "small moving object") reaches DNa02 only through
  the HS cells and weakly: front of the right eye, T4a+T5a → DNa02/R 53 Hz pure / 10 game,
  DNp15/R 97 / 80; centre → DNa02/R 10 / 20; back → 0 (HSE only). HS cells are retinotopic here
  (HSN for front and centre, HSE for the back).
* Injecting the lobula input layer instead (15 types, right eye, 120 Hz) shows no size
  selectivity in this model: every LC responds more to the whole field than to a patch. Whole-eye
  drive of single types: T2 → LC18 58 Hz and LC4 82; TmY3 → LC4 125; Tm4 → LC4 35; T3 → LC11 52,
  LC21 14; Tm12 → LC25 120, LC11 15; Tm6 → LC22 22, LC11 7; Tm20 → LC16 11, LPLC1 13; Tm5Y →
  LPLC1 21, LPLC2 11, LC20a 25; 19-column patches of the same types give ≤ 3.4 Hz in any LC.
* `LC10a`, the lure detector, never exceeds 1.3 Hz from any columnar drive: its excitatory
  columnar input (Tm5Y, 49 synapses per cell) is outweighed by inhibitory feedback from `TuTuA_2`
  (72 per cell) and `AOTU042` (52).

### 5.7 The resulting design

* **Columns from the retina (wiring + hand-built correlator).** Each facet is mapped to the nearest
  medulla column of its eye; each T4/T5 cell to its inferred home column. A Hassenstein-Reichardt
  correlator (Hassenstein & Reichardt 1956) on the *change* of brightness at each facet — the
  change now at a facet times the recent change (60 ms low-pass) at the neighbour the motion came
  from, minus the mirror term — gives one signed motion signal per axis; the positive part drives
  the matching subtype (a/b for azimuth, c/d for elevation), brightening edges drive T4 and
  darkening edges T5, responses are pooled with half weight from neighbouring facets (T4/T5
  dendrites span a few columns), normalised per column and turned into rates of up to 120 Hz
  (`gain_hz`; rates below 2 Hz are dropped). Only moving edges produce anything; the interior of an
  object gives nothing, and opposite subtypes of a column are never driven together. Everything
  from the T4/T5 spikes onward — HS, H2, VS, LPLC2, DNa02, DNp15 — is the connectome.
* **Optomotor steering read from `DNp15`.** The decoder adds `0.35 · (DNp15/R − DNp15/L) / 60` to
  the yaw command (besides `DNa02`, `DNg13`, `DNa01`, `DNa03`), so front-to-back motion on the
  right eye turns the fly right, following the motion. A striped drum on the wall (up to 48
  stripes, rotating at up to 6 rad/s) is the classic stimulus, and the "optomotor" checklist item
  passes when the fly turns with it.
* **Efference copy (hand-built).** While the fly turns on purpose, the columnar motion signal is
  scaled by `max(0.15, 1 − 2.5·|turn command|)`, so its own turn does not trigger the reflex that
  would cancel it (Kim, Fitzgerald & Maimon 2015 showed exactly this in HS cells).
* **Feature detectors stay** for `LC4`/`LPLC2` (looming) and `LC10a`/`LC11` (small object),
  because T4/T5 do not reach LC10a at all and reach LPLC2 too weakly for a reliable escape. The
  probe's LC4-input injection (T2, TmY3, Tm4 per column) was not adopted: LC4 has no direction
  selectivity, so it would fire for any large change in the image.
* **Escape.** The game starts a jump on the first giant-fibre spike that reaches the body. With
  the feature detectors gated at 60 °/s this is selective in practice; the probe's caveat stands
  that a rate threshold (e.g. DNp01 ≥ 60 Hz over 50 ms) would be needed if looming were ever read
  from the columns alone, where the wide-field flash of a large object gives DNp01 45-95 Hz and a
  radius-3 patch 6.7 Hz.

---

## 6. Courtship, hearing, wind and walking (the `courtship-mechano` probe)

All numbers: pure and probe-game, 100 Hz Poisson stimuli, 500 ms (100 ms discarded), seed 0,
starter `fly_brain.py`. Motor pools are the `vnc_motor` neurons grouped by name and neuromere
(`common.py`): wing power (DLMn, DVMn), wing steering (b, i, iii, hg, tp, ps, MNwm), jump (TTMn,
STTMm), haltere, neck, abdominal, and leg T1/T2/T3 (the remaining thoracic motor neurons).

### 6.1 Pheromone to pC1

The male's tarsal taste neurons sense the female's cuticular hydrocarbons when he taps her. A
wiring census (`census.py`, `task1`) ranked leg and wing gustatory types by their excitatory
two-hop score to the 148 `pC1` neurons (sum over relays of synapses in × synapses out): LgLG1a
(136 neurons) is first, almost entirely through the ascending neuron `AN05B102c`, then WG4,
LgLG1b, WG3, SNxx23, LgLG6 (via AN03A008), LgLG2, LgLG4. But `AN05B102c` makes only 91 synapses
onto 6 pC1 neurons (27 onto the best one), against a median of 514 excitatory input synapses per
pC1 cell, and in simulation:

| stimulus | config | pC1 | pIP10 | what does fire |
|---|---|---|---|---|
| LgLG1a at 100 Hz | pure / game | 0.0 / 0.0 | 0 / 0 | AN05B102c 200 / 185 Hz, MDN 20 / 23, DNge050 62-82, DNge053 51-69; wing power 38 / 33 Hz (a runaway follows in pure: 543k after) |
| LgLG1a at 200 / 400 Hz | game | 0.0 / 0.2 | 0 | |
| LgLG1a + LgLG1b + WG3 + WG4 at 100 Hz | pure / game | 0.1 / 0.0 | 1.2 / 0 | AN05B102c 331 / 290 |
| same four at 300 Hz | game | 0.2 | 0 | |
| AN05B102c direct at 100 / 300 / 600 Hz | game | 0 / 0 / 0 | 0 | |
| LgLG4, LgLG2, LgLG6, WG3, SNxx23 at 100 Hz | both | 0 | 0 | LgLG4 → DNg103 116 / 100; LgLG6 → DNa02/L 20 / 17.5 |
| every 1-hop excitatory input type of pC1 at 100 Hz, one at a time (25 types) | game | at most 4.2 Hz (AN08B074; with pIP10 42, pMP2 38) | | LC16 → pC1 2.0, pIP10 6 |
| LgLG1a at 100 Hz with every synapse onto pC1 multiplied by 2 / 4 / 8 | game | 0.3 / 6.8 / 33.6 | 0 / 7.5 / 51 | at ×8: DNp13 12.5, oviIN 145, ps1 MN 72, hg1 MN 71 |

The pheromone route to pC1 exists in the wiring but is about eight times too weak in this model.
The game therefore drives `LgLG1a,LgLG1b` at 60 Hz on foreleg contact (the wiring part, visible
as `AN05B102c` and `MDN` activity) **and, hand-built, drives `prefix:pC1_` itself at 60 Hz for
2.5 s after each tap** (`COURTSHIP_SPEC`, labelled "courtship arousal" on screen). Everything
downstream of pC1 is wiring.

### 6.2 pC1 to song

`task2`, `task2b`:

| stimulus | config | pIP10 | DNp13 | wing motor neurons | pools |
|---|---|---|---|---|---|
| all pC1 at 80 Hz | pure | 97.5 | 31 | ps1 MN 74, hg1 MN 49, DLMn c-f 36, DVMn 14 | wing power 25, wing steer 28, abdominal 19; also oviIN 241, AOTU019 91, DNg33 255 |
| all pC1 at 80 Hz | game | 104 | 47.5 | ps1 68, hg1 64, hg3 115, hg4 80, DLMn c-f 28 | wing power 17.5, wing steer 26.5, abdominal 13 |
| all pC1 at 40 / 150 Hz | game | 61 / 157 | 14 / 108 | | wing power 7 / 42 |
| pC1_14a, pC1_5b, pC1_7b at 80 Hz | game | 52.5 | 56 | | pMP2 34; no oviIN drive |
| pC1_16a, pC1_16b at 80 Hz | game | 0 | 0 | | aSP22 52.5, MN9 20 |
| pIP10 alone at 80 / 150 Hz | game | | 0 | hg3 105 / 131, hg4 79 / 99, ps1 68 / 80, hg1 51 / 58, MNwm35 41 / 40, DLMn c-f 22 / 29 | wing steer 22 / 27, wing power 11 / 16, legs 0 |
| pC1 at 80 Hz with pIP10 silenced | game | (fires, muted) | 51 | hg3 30, hg4 14, DLMn c-f 0.3 | wing power 1.1, wing steer 11 |
| pC1 at 80 Hz with DNp13 silenced | game | 100 | (muted) | unchanged | wing power 15 |

So `pC1 → pIP10 → wing steering and power motor neurons` is the song pathway in the wiring
(pIP10's strongest inputs: aIPg7 1,280 synapses, pC1_14a 565; its outputs: AN08B061 2,113, dPR1
1,112, IN12A030 979, the TN1a cells), and silencing pIP10 removes the wing-power drive. `vPR6`
is not reached by pC1 (0 Hz); driven directly it recruits dMS5, ps1 and hg1. The decoder reads
`pIP10` (song, one wing out toward the female) and `pC1` (courtship mode); the abdominal bend
near the female is hand-built.

### 6.3 Hearing

`task3`; readouts in Hz:

| stimulus | config | DNp01 | AMMC-A1 | DNp11 | pC1 / pIP10 / vPR6 | after |
|---|---|---|---|---|---|---|
| JO-A at 100 Hz | pure / game | 0 / 0 | 0 | | 0 | 0 / 4.9k |
| JO-B at 100 Hz | pure / game | **71 / 64** | 44 / 35 | 65 / 50 | 0 | 0 / 0 |
| JO-A + JO-B at 100 Hz | pure / game | 30 / 32.5 | 12 / 10 | 19 / 14 | 0 | 0 / 0 |
| subclass:auditory at 100 Hz | pure / game | 32.5 / 35 | 22 / 21 | | 0 | 0 |
| JO-A + JO-B at 200 Hz | pure / game | 17.5 / 45 | | | pIP10 7.5 / 0 | 548k (runaway) / 37k |

The auditory Johnston's-organ neurons reach the giant fibre (JO-B1_a makes 541 synapses onto
the two DNp01 cells; JO-B at 100 Hz drives DNp01 at 64-71 Hz), which is the acoustic startle real
flies show, and nothing in the courtship circuit (pC1, pIP10, vPR6 all 0 Hz). The game's "clap"
drives `prefix:JO-B` at 100 Hz (`SOUND_SPEC`): the B channel alone reaches the giant fibre at
64-71 Hz, whereas A and B together give only 30-32 Hz. The female's song is not modelled (the
male's hearing input is zeroed while she is present).

### 6.4 Wind

`task4`; the wind populations are `JO-CA1,JO-CA2,JO-CL,JO-CM,prefix:JO-ED,prefix:JO-EV` (203
neurons left, 132 right); readouts in Hz, pure (game in brackets):

| stimulus | DNg62 L / R (aDN1, grooming) | DNge078 L / R (aDN2) | MDN L / R | DNa03/L | steering DNs (DNa02, DNa01, DNg13, DNa04-06, DNb01-02) | after |
|---|---|---|---|---|---|---|
| left antenna at 100 Hz | 178 / 185 (138 / 148) | 128 / 133 (98 / 108) | 12.5 / 7.5 (14 / 7.5) | 0 | all 0 | 140k (32k) |
| right antenna at 100 Hz | 188 / 150 (140 / 103) | 143 / 113 (105 / 73) | 63 / 59 (65 / 56) | 25 (12.5) | DNa02/L 2.5, DNa05/L 7.5, rest 0 | 146k (0) |
| both at 100 Hz | 188 / 195 (145 / 148) | 148 / 148 (110 / 110) | 44 / 39 (16 / 14) | 5 (2.5) | 0 | 143k (40k) |
| left antenna at 50 Hz | 180 / 178 (130 / 135) | 130 / 123 (105 / 100) | 41 / 28 (65 / 55) | 10 (10) | 0 | 138k (8.8k) |

A left/right scan of every descending type found no DN whose side follows the side of the wind:
the asymmetric ones (DNde006, DNge045, DNp101 fire on the right only) do so for left, right and
bilateral wind alike, i.e. they reflect a wiring asymmetry of this one brain, not wind direction.
What steady antennal deflection does reach, at 50 Hz already, is the grooming pair `DNg62`/
`DNge078` at near-dust levels and `MDN` (backing): in this model sustained JO-C/E input is a
"something on my antennae" signal, not a compass (Suver et al. 2019 recorded wind-direction
tuning in central neurons downstream of these cells; the LIF model does not reproduce it). The
game therefore keeps wind weak — `WIND_MAX_HZ` = 20 Hz on the two JO-C/E populations, scaled by
the airflow at each antenna (the code notes that above ~30 Hz the same neurons trigger grooming;
20 Hz itself was not measured in the probe, whose lowest point was 50 Hz) — so the on-screen
readout shows which antenna the wind hits without the fly grooming or backing up, and heading
upwind during odour search is hand-built (section 4.8).

### 6.5 Descending neurons to the legs

`task5`, `task5b`: each candidate DN type at 80 Hz for 400 ms, rates of the leg motor pools in
Hz (pure / game):

| DN | role in the decoder | leg motor neurons, all (T1 / T2 / T3) | other pools |
|---|---|---|---|
| DNp09 | forward | 1.2 / 1.1 (0.4 / 1.1 / 2.1 pure) | wing power 16 / 12, neck 11 / 10; DNg33 132 / 123 |
| DNg100 (BDN2) | forward | 1.3 / 1.3 (0.6 / 0.8 / 2.5) | |
| MDN | backward | 0.8 / 0.7 (0 / 1.0 / 1.5) | DNg64 33 |
| DNa02 (both sides, or left only) | steering | ≤ 0.4 in any pool (T3 0.4 pure) | neck 1.9 |
| DNa01 | steering | ≤ 0.2 | |
| DNg13 | steering | ≤ 0.6 (T1 0.6) | MN9 8 / 3 |
| DNa03 | steering | 0.8 / 0.6 | wing power 37 / 25 |
| DNa04, DNa05, DNa06 | | ≤ 0.8 (DNa06 T3 0.8 pure) | |
| DNg60, DNg74_a, DNb01, DNb02 (inhibitory) | halt | 0 (only they themselves fire) | |
| AN19A018 | halt | 0.1 | wing power 52 / 29 |
| aSP22, pMP2 | | 0.9 / 0.7, ≤ 0.2 | aSP22: MN9 22 |
| MDN at 200 / 400 Hz | | 2.7 / 3.9 (game) | |
| DNg100 at 200 Hz | | 4.2 (game) | |
| all 1,314 descending neurons at 80 Hz | | 7.7 (11.6 / 6.8 / 4.6, game) | wing power 165, pIP10 55 |
| gain 0.8 instead of 0.65, DNp09/DNg100/MDN at 80 Hz | | 0.8 / 2.5 / 2.3 | |
| gain 1.0 | | 2.6-5.3, brain at 170-320k spikes/s | |
| gain 1.3 | | 6.5-9.1, brain at 480-540k spikes/s (runaway) | |

Single descending neurons move the leg motor neurons by only a few spikes per second in this
model, whatever the rate or the gain, and adding an inhibitory "brake" DN on top of DNg100
changes nothing (leg pool 1.0-1.7 Hz in every combination). The leg motor circuits of the nerve
cord (central pattern generators, proprioceptive loops) are not something a rate-driven LIF
reproduces. The **motor decoder is therefore hand-built**: it reads the descending neurons that
the literature and the wiring assign to each behaviour and turns their rates into drives
(`game.py`, `MotorDecoder`):

```
forward  = 0.3 n(DNp09/50) + 0.25 n(DNg100/50) + 0.15 n(DNge053/50) + 0.15 n(DNge050/50) + 0.15 n(DNg97/50)
yaw      = (DNa02R−L)/50 + 0.5 (DNg13R−L)/100 + 0.25 (DNa01R−L)/50 + 0.2 (DNa03R−L)/50 + 0.35 (DNp15R−L)/60
backward = n(MDN/60);  halt = 0.6 n(DNg60/50) + 0.4 n(DNg74/50) + n(AN19A018/50)
feed = n(MN9/50);  groom = n(max(DNg62, DNge078)/100);  song = n(pIP10/40);  court = n(pC1/30)
```

(`n(x)` clips to 0..1; each drive is low-pass filtered with a 0.1-0.4 s time constant.) A
behaviour wins when its drive exceeds a threshold (backward 0.25, feed 0.3, groom 0.35) and keeps
going until it falls to half; the giant fibre overrides everything. The body is a kinematic
model: 14 mm/s at full forward drive, 8 mm/s backward, 300 °/s turn rate, a 22 mm hop in 0.16 s,
with short inertia. At start-up the decoder also measures, from the wiring, how many synapses
each of its DNs sends to leg, wing, neck and abdominal motor neurons within two hops, and shows
that on screen as the reason a DN is read as "forward" or "turn". The weights above are not
measured from anything.

---

### 6.6 Genetics: fruitless, doublesex, and the driver lines (v2.2)

The genome cannot be loaded into the model; what can is gene expression per cell type, and the
MaleCNS v1.0 file carries two kinds of it. The team registered light-microscopy images of
*fruitless* and *doublesex* expression to the EM volume and marked matching neurons as expressing
the gene with high or low confidence, and compared every cell type with a female connectome to
label male-specific and sexually dimorphic neurons (Sexual dimorphism in the complete Drosophila
male CNS connectome, Cell 2026). The counts in the kit's file match the paper exactly:

| label | neurons | cell types |
|---|---|---|
| *fru* (high 2,611 + low 1,989 + both 258) | 4,858 | 870 |
| *dsx* (high 138 + low 16 + both 258) | 412 | 119 |
| both | 258 | |
| male-specific (incl. potentially) | 1,420 | |
| sexually dimorphic (incl. potentially) | 948 | |

The transmitter of every neuron (acetylcholine 59.9 % of synapses, GABA 21.3 %, glutamate 16.1 %,
histamine, serotonin, octopamine, dopamine; 10,900 neurons "unclear", counted excitatory) was
predicted from synapse appearance; making a transmitter is the work of a few genes (ChAT/VAChT,
Gad1/VGAT, VGlut, Hdc, Trh/SerT, ple/DAT, Tdc2/Tbh), which `genetics.py` names and links to
FlyBase, because they are the genes the model's one physiological rule rests on.

**What the courtship circuit expresses.** Every pIP10 song neuron is *fru*+ and male-specific,
all 148 pC1 command neurons are *dsx*+ (88 also *fru*+) and male-specific, all 275 LC10a chase
neurons are *fru*+, and 78 % of the PAM dopamine neurons carry a low-confidence *fru* label; MN9,
the giant fibre, MDN, the Kenyon cells and PPL1 carry no label. So the classic result that
*fruitless* males do not court can be run as a prediction. Measured in the game profile (seeds 0
and 1; pC1 driven at 60 Hz for 500 ms; `gene:fru` = all 4,858 neurons' output blocked):

| readout | intact | *fru* neurons silenced | *dsx* neurons silenced |
|---|---|---|---|
| pC1 | 63 Hz | 62 | 60 |
| pIP10 | 79 | 45 (fires, output blocked) | 0 |
| ps1 song motor neurons | 61 | 0 | 0 |
| hg1/hg2 song motor neurons | 57 | 0 | 0 |
| DLMn wing power motor neurons | 22 | 0 | 0 |
| sugar → MN9 (control) | 27 | 34 | |
| looming → DNp01 / TTMn (control) | 292 / 60 | 286 / 53 | |
| vinegar → MBON14 (control) | 54 | 52 | |

These are the five `GENETIC` experiments in `experiments.py` (ranges set from these numbers). In
the game, silencing `gene:fru` also removes the chase, because LC10a is *fru*+, which matches the
mutant phenotype (no courtship at all); the decoder only reads neurons whose output is not blocked.

**Driver lines.** NeuronBridge (Janelia) matches expression images of split-GAL4 and GAL4 lines to
EM neurons by colour-depth search, and its public data covers the MaleCNS brain and nerve cord
(libraries `FlyEM_Male_CNS_Brain_v0.9` and `_VNC_v0.9`). Checked against the kit's v1.0 file for
DNp01, pIP10, LC10a, MBON01, MDN, DNa02 and PAM01: the same body IDs exist with the same cell-type
names (7 of 7; T4a has no entry, as optic-lobe intrinsic types are not in the colour-depth
libraries). `genetics.NeuronBridge` fetches, for a population, the lines that match a sample of its
neurons (each match file lists ~1,600 line images with scores; lines matching several of the
sampled neurons rank first, split-GAL4 lines are flagged as the clean ones), and for a line the
MaleCNS neurons its best images match, with NeuronBridge's cell-type name next to the kit's so a
version difference is visible. Everything is cached on disk; without internet the lookups fail with
one message and the rest of the kit is unaffected.

**What this is not.** Two transcription factors and a transmitter identity are all the genes this
data can speak for. Per-cell-type ion channels, receptors (which would fix the sign of the
"unclear" neurons and the cases where glutamate excites) and neuromodulator receptors would need
single-cell expression atlases matched to these cell types, which is mature only for the visual
system, the olfactory projection neurons and the mushroom body; that is the next level, and it
would ship as a separate profile judged by the validated experiments.

## 7. The genome as a wiring recipe (v2.3)

A genome of some 140 million letters cannot list the fly's 90 million synapses; it holds rules
that build the wiring during development. `wiring.py` asks how much of the fly's behaviour lives
in such rules, using the connectome itself as the only source: the rules are learned from the
wiring, a new fly is grown from the rules alone, and the validated experiments are run on it.

**Rules.** For every pair of *(cell type, side)* groups, the number of connected neuron pairs and
the log-normal shape of their synapse counts. MaleCNS v1.0 has 23,078 such groups and 1,704,511
connected pairs, i.e. 5.1 M numbers stand in for 6.3 M connections. **Growing** keeps every neuron
and draws its connections afresh within the rules: a dense pair (most of two small groups wired
together) is decided connection by connection with the pair's probability, a sparse pair is
sampled, synapse counts come from the pair's distribution, and duplicates merge. A grown fly has
the same number of connections and synapses (6.26 M / 91 M against 6.29 M / 90 M) and shares
36 % of its individual connections with the real one; the rest are new neuron-to-neuron pairings
of the same types. **The bottleneck** approximates the rule matrix at rank K with a randomized SVD
(each group gets a K-number output code and a K-number input code; a rule is their product),
re-thresholded to the original number of rules and rescaled to the original number of connections.
**Class rules** use only superclass:class and side (125 groups, 2,665 rules).

**What survives** (game profile; each grown fly tested with all eleven validated experiments;
`fly-brain --genome-sweep`):

| experiment | real | type, seed 1 | type, seed 2 | bottleneck 256 | bottleneck 64 | bottleneck 16 | class |
|---|---|---|---|---|---|---|---|
| Silence (no input) | ok | ok | ok | ok | ok | ok | ok |
| Sugar on the mouthparts | ok | ok | 3/4 | 1/4 | 1/4 | 1/4 | 1/4 |
| Bitter taste | ok | ok | ok | 1/2 | 1/2 | 1/2 | 1/2 |
| Sugar + bitter together | ok | ok | ok | ok | ok | ok | ok |
| Something looming on the right | ok | ok | ok | 0/2 | 0/2 | 0/2 | 0/2 |
| Dust on the antennae | ok | ok | ok | 0/2 | 0/2 | 0/2 | 0/2 |
| Smell of vinegar | ok | 3/4 | 3/4 | 0/4 | 2/4 | 0/4 | 0/4 |
| Bitter taste → punishment dopamine | ok | ok | ok | 1/2 | 1/2 | 1/2 | 1/2 |
| A loud sound | ok | ok | ok | 0/1 | 0/1 | 0/1 | 0/1 |
| Courtship command | ok | ok | ok | ok | ok | 0/2 | 0/2 |
| Wide-field motion, right eye | ok | 3/4 | ok | 1/4 | 1/4 | 1/4 | 1/4 |
| **experiments passed** | **11** | **9** | **9** | **3** | **3** | **2** | **2** |
| connections shared with the real fly | 100 % | 36 % | 36 % | 13 % | | | 0.4 % |
| numbers in the genome | 12.6 M | 5.1 M | 5.1 M | 11.9 M | 3.0 M | 0.8 M | 8 k |

Three findings:

1. **Nine of eleven reflexes live in the type-level rules.** Feeding, bitter suppression, looming
   escape, grooming, the smell code, punishment dopamine, the startle and the courtship command
   all work in a fly whose neurons were never individually wired, and the two failures are
   marginal: MBON11's specific Kenyon-cell input (0 Hz instead of 1-60) and, in one individual,
   the Fudog taste interneuron or a 27 Hz leak into the wrong DNa02 during right-eye motion. The
   two individuals differ in which marginal readout they miss, which is what individual
   variation looks like here.
2. **Class-level rules carry nothing.** With cell-type identity removed every reflex is gone;
   the sugar relay, the giant fibre and the song neuron sit silent because their inputs are
   spread over a whole class.
3. **The rule matrix is not low-rank.** A rank-256 code holds *more* numbers than the rule table
   (11.9 M against 5.1 M) yet keeps only three experiments, and a rank-64 code the same three
   (courtship survives because its 148 command neurons in many types form a broad block; feeding
   dies because LB3b → GNG232 → MN9 is one specific rule per link). Specificity, not volume, is
   what this genome has to encode, which is why the SVD is a weak bottleneck and why the genomic
   bottleneck literature learns its codes with a network rather than a factorisation. A learned
   bottleneck is the natural next experiment on this data.

What growth destroys, by construction, is neuron-level structure inside a type: the retinotopic
column-to-column wiring of the optic lobe (the columnar looming path and the LPLC2 outward
arrangement of section 5.3), the Kenyon cells' individual glomerular samples, and any left/right
pairing finer than the side label. The experiments that inject T4a/T5a or LC4/LPLC2 as
populations do not test that structure; a grown fly's eyes work in the game because the retina
still drives the real columns, but what those columns feed is rewired.

## 8. The genes as a parts list (v2.4)

Section 6.6 reads which genes a neuron expresses and section 7 grows the wiring from rules. The
step between them is what the genes *build*: the published model gives every neuron the same
machine (a leaky integrate-and-fire unit whose one per-neuron property is the sign of its
transmitter), and real neurons are not like that. `parts.py` changes the model in the two places
where the literature is solid enough to say how, and leaves a table for the rest.

**Three transmitters act only through slow receptors.** Every dopamine receptor in *Drosophila*
(Dop1R1, Dop1R2, Dop2R, DopEcR), every octopamine receptor (Oamb, Octα2R, Octβ1R-3R) and every
serotonin receptor (5-HT1A, 1B, 2A, 2B, 7) is G-protein-coupled (Karam et al. 2020; El-Kholy et
al. 2015; Blenau & Thamm 2011); none opens an ion channel. A spike in one of the 399
dopaminergic, 165 octopaminergic and 415 serotonergic neurons of MaleCNS therefore cannot make
a fast synaptic potential, yet the published model, and the game profile for octopamine and
serotonin, treat them as ordinary excitatory neurons (the game profile already muted the
dopamine neurons' fast synapses by hand, section 4). With the parts list on, all 979 lose their
fast synapses (79,183 connections onto 30,157 targets) and each spike instead adds to a *tone*
on its targets: one unit per ten synapses, decaying with a time constant of 0.5 s (dopamine;
Cohn, Morantte & Ruta 2015 see dopamine transients of about a second in the mushroom body),
1 s (octopamine) or 2 s (serotonin). The tone scales the target's synaptic input,
`gain = 1 + a · level / (level + 5)`, saturating at +30 % for dopamine and serotonin and +50 % for
octopamine (Suver, Mamiya & Dickinson 2012 measured about that much gain on the lobula-plate
tangential cells from octopamine during flight; Dacks et al. 2009 saw serotonin enhance
antennal-lobe projection-neuron responses over seconds). The sign is one choice per transmitter,
and that is the model's honest gap: each target's receptors decide whether a tone excites
(Gs-coupled Dop1R1, Oamb, 5-HT7) or inhibits it (Gi-coupled Dop2R, 5-HT1A), and receptor
expression per cell type is not in this data. The dopamine neurons keep their role in plasticity
(section 4.5) unchanged; silencing or modulating a modulatory population silences or scales its
tone as it would any output.

**Five groups of optic-lobe cell types do not spike.** Photoreceptors R1-R8 (6,091 neurons here;
Hardie & Raghu 2001), the lamina monopolar cells L1-L5 (8,884; Laughlin & Hardie 1978; Zheng et
al. 2006), the medulla columnar inputs to the motion detectors Mi1, Mi4, Mi9, Tm1, Tm2, Tm3, Tm4
and Tm9 (14,358; Behnia et al. 2014; Yang et al. 2016; Arenz et al. 2017), T4 and T5 themselves
(13,585; Gruntman, Romani & Reiser 2018) and the tangential cells HS, VS and CT1 (44; Haag &
Borst 1996; Schnell et al. 2010; Meier & Borst 2019) signal with graded potentials. With the
parts list on these 42,962 cells release transmitter in proportion to their depolarisation: a
cell held at the spike threshold releases 300 quanta per second, one at half threshold 150, and
each quantum is delivered to the targets exactly as a spike's worth of transmitter would be.
There is no threshold, no reset and no refractory period, so a graded cell transmits inputs a
spiking model would drop, and its output saturates instead of racing. What this model does not
capture is tonic release: the real OFF pathway signals by *reducing* a resting release, and here
a cell at rest releases nothing. The retina still drives T4/T5 as forced events (section 5.2), so
the game's vision is unchanged upstream; what changes is everything the T4/T5 and HS/VS cells
feed.

**A table for the rest.** `parts.CELL_PARAMS` takes per-type overrides of the spike threshold and
of the graded flag in the ordinary population-spec language (`--part "class:Kenyon_Cell:theta=10"`
from the terminal), so measured values, from patch recordings or one day from ion-channel
expression per cell type in the Fly Cell Atlas, can be dropped in without touching the model. It
ships empty: no per-type threshold in the literature is solid enough to hard-code.

**What it does to the validated experiments** (game profile, every experiment of sections 2, 4-6
and 6.6, parts list off against on; `fly-brain --profile game --parts`):

| experiment | parts off | parts on |
|---|---|---|
| Silence (no input) | ok | ok |
| Sugar on the mouthparts | ok (MN9 50 Hz) | ok (MN9 73 Hz) |
| Bitter taste | ok | ok |
| Sugar + bitter together | ok | ok |
| Something looming on the right | ok (GF 295 Hz) | ok (GF 305 Hz) |
| Dust on the antennae | ok | ok |
| Smell of vinegar | ok (MBON11 1.0 Hz) | **MBON11 0 Hz** (range 1-60), the other three readouts ok |
| Bitter taste → punishment dopamine | ok (PPL101 78 Hz) | ok (PPL101 113 Hz) |
| A loud sound | ok | ok |
| Courtship command | ok (DNp13 39 Hz) | ok (DNp13 5 Hz, at the edge) |
| Wide-field motion, right eye | ok (HS 443 Hz, DNa02 left 0 Hz) | ok (HS 300 Hz, DNa02 left 20 Hz, at the edge) |
| the five genetic experiments | ok | ok |
| **passed** | **16 / 16** | **15 / 16** |

Fifteen of sixteen survive a change that removes 79,000 fast synapses and rewrites the output
rule of a quarter of the brain, which says the validated reflexes do not depend on the
monoamine neurons' fast synapses or on the optic lobe's spiking. The one miss is the marginal
readout of the vinegar experiment: MBON11's Kenyon-cell drive gives 1.0 Hz in the standard model,
the lower edge of its range, and 0 Hz with the parts on (its PPL1 dopamine tone is absent without
bitter taste, and its input gain is unchanged). Section 8.2 finds the cause, APL, and v2.6's change. Three readouts move to the edges of their ranges,
all in the direction the parts predict: the graded HS cells saturate at 300 events/s instead of
firing at 443 Hz and transmit their sub-threshold contralateral input (DNa02 left 20 Hz), and the
courtship command loses the octopaminergic and serotonergic fast drive that DNp13 was getting.
The tones themselves are silent in these experiments because none of them drives a modulatory
population for long; in the game a loud sound, bitter taste or a busy scene raise them (a busy
second of sugar, motion, smell and sound leaves the octopamine tone at 66 % of its full effect
on 15,000 targets, dopamine at 35 %, serotonin at 41 %).

**Cost.** The graded cells' release, the tones' deposit and the gain applied to arriving input
are compiled kernels in the numba backend and plain array operations in NumPy, with the same
float32 arithmetic in the same order, so both backends stay spike-identical (the test suite
checks it with every modulator and graded cell in play). A busy second of brain time (sugar,
motion, smell and sound at once) costs 1.15 s with the parts on against 0.73 s off in the
compiled backend, a third more, mostly because the graded cells emit a third more events (316,000
against 235,000 a second under a typical game load). In the game that means a real-time factor of
0.8-0.9 under heavy stimulation on this machine, against 1.0 with the parts off; `--fast` (a 1 ms
step) restores real time. In NumPy the parts add about 60 %.

### 8.1 What Virtual Fly Brain adds (v2.5)

The connectome names every neuron with a MaleCNS type string; the rest of fly neuroscience is
keyed by the classes of the FlyBase anatomy ontology, FBbt (Costa et al. 2013), which is what
Virtual Fly Brain (VFB; Court et al. 2023) indexes: a curated definition per class, a place in an
`is_a` tree, the transmitter the literature has established, and every other data set that saw the
same cell type. `vfb.py` joins the two vocabularies offline and uses the join in four places.

**The join.** `tools/build_vfb_data.py` reads the ontology's own OBO release (2026-07-09, CC-BY
4.0) and matches each of the 11,751 kit type names against the labels, VFB symbols and EXACT
synonyms of the non-obsolete adult neuron classes, in that order of preference, with three
tie-breaks (prefer the class whose label ends with the code, so `L1` is the lamina monopolar cell
and not a lateral-horn neuron with the same symbol; prefer the adult class; never a female-specific
one) and no guessing: a name that still matches two classes is left unresolved. RELATED synonyms
are never used, because they are old names that collide (Tm36 is a RELATED synonym of TmY21, Li28
of Li16). Comma-joined names map to every member and a `_a`/`_b` suffix falls back to its stem's
class, marked *coarse*. That resolves 9,434 types (86.2 % of the typed neurons)
without touching the network. The names the OBO lacks are VFB's own `name_in_male-cns` synonyms,
which live only on VFB's server: a one-time harvest through the VFB connector (`search_terms` with
one row per matching synonym, then `get_term_info` to check that the class carries the exact
MaleCNS name) adds 256 more types (9,436 neurons: the 3,377
photoreceptors `R1-R6`, the 745 interommatidial bristle neurons `BM_InOm`, `KCab-m`, the LC10c
subtypes, `MN9`, `pIP10`, `GNG232`, ...; for the 40 `pC1_*` subtypes and a dozen sensory groups
that no class names, the class is read off one of the type's own neurons on VFB, found by its
bodyId, and marked coarse). The same check on 363 of the offline matches (every
EXACT-synonym match, every type whose curated transmitter disagrees with the prediction, and random
samples of the symbol and stem matches) confirmed 352 and contradicted 11: a
`_b` type whose stem class lists its MaleCNS members without it (`PS008_b`, `DNg36_b`, `CB1287_b`,
...) is left unresolved, and where VFB shows the MaleCNS name on a different class than the OBO
match (hemibrain and MaleCNS reused a name for different cells: `SMP053`, `SLP305`, four `LHAV`
types) the map uses that class, each checked by hand (`tools/vfb_overlay.json` records it). One
in seven of the sampled stem matches is contradicted this way, which is why stem matches never
drive the model (below). The result is `data/fbbt_map.json.gz` (9,690 types, 92.0 % of the typed neurons;
`tools/vfb_overlay.json` holds the harvest so the build is reproducible) and
`data/fbbt_tree.json.gz`, the 9,833 classes above them with labels, parents, symbols
and short definitions. The unresolved remainder is mostly names MaleCNS coined and no ontology
class carries yet (`TmY9a`, `Tm38`, `MeTu3c`, most `SNta`/`SNpp` sensory groups, the `pC1_*`
subtypes) and the 11,916 untyped neurons.

What the join gives: a selector, `fbbt:<class>`, that takes the ontology's `is_a` closure, so
`fbbt:lobula columnar neuron` is every LC type the kit has (3,652 neurons), `fbbt:adult descending
neuron` every DN (1,221), `fbbt:dopaminergic neuron` every cell the ontology calls dopaminergic
(1,577), `fbbt:adult Kenyon cell` all 3,528 Kenyon cells, composable with everything else
(`fbbt:adult descending neuron&nt:gaba`); an ontology search in the Neuron lab; and, in a neuron's
popover, what its type *is*: the class and its definition, the anatomical parent chain (each parent
a click away as a population), the lineage, the peptides the class is known to express, the
transmitter the literature asserts, and a link to VFB.

**Curated transmitters.** The ontology asserts a transmitter for a class by making it a subclass of
*cholinergic neuron*, *GABAergic neuron*, and so on; VFB's `get_known_neurotransmitters` reads the
same assertions (a sample of 65 classes checked through the connector agreed with the offline
reading in 65 of 65). Two kinds of class carry them: the literature-curated classes (FBbt ids below
2000 0000: the transmitter comes from immunostaining, driver lines or transcriptomics cited in the
class definition) and the systematic connectome-derived classes (`FBbt:2xxxxxxx`, one per hemibrain,
FlyWire or MANC type, whose transmitter is that data set's own prediction). Against the MaleCNS
prediction, the literature agrees for 7,400 of the mapped types with a curated transmitter and
differs for 213; 184 types whose prediction is "unclear" get one. With the parts
list on, the literature's word wins where the parts model cares (`PartsList(curated="modulators")`,
the default, which uses the literature-curated classes only; `vfb.transmitter_overrides` has the
rules):

| what the curated class says | neurons | the parts list does |
|---|---|---|
| Mi15 is cholinergic *and* dopaminergic (Davis et al. 2020) | 1,151 | keeps its fast synapses and adds a dopamine tone on its targets |
| the DPM neuron (predicted dopamine) is GABAergic and serotonergic; PPL203 (predicted serotonin) is dopaminergic and GABAergic; OA-ASM3 (predicted serotonin) is octopaminergic; one vMS17 (predicted octopamine) is dopaminergic, GABAergic and serotonergic | 7 | the tone becomes the literature's modulator (the first of several, for vMS17), and a co-released GABA gives fast inhibitory synapses |
| DNg34 (glutamate) and DNg66 (acetylcholine), predicted octopaminergic, co-release a fast transmitter | 3 | the octopamine tone stays and the co-released transmitter's fast synapses are added |
| LHPV6q1, PRW068 and one aMe8 (predicted serotonin) are cholinergic | 5 | no tone; fast excitatory synapses |
| MeVCMe1 and DNd03 (octopamine) and LPsP (dopamine) are modulatory, predicted fast | 8 | a tone is added and the predicted fast synapses are kept (under `curated="all"` they are removed) |
| OA-ASM2, FB6H, FB7B, PAL03 and vMS16, predicted "unclear", are octopaminergic or dopaminergic | 10 | a tone and no fast synapses, like any modulatory neuron |
| DNd02 and one vMS17, predicted "unclear", release a modulator and a fast transmitter | 3 | a tone and fast synapses |
| "unclear" types with a curated fast transmitter: TmY14 (91 neurons), LHAV4d1, CEM, aMe8 and two more | 114 | the sign of their fast synapses |
| under `curated="all"` only: 16 more "unclear" types whose only class is another connectome's type, predicted dopaminergic or serotonergic there (SMP143, ATL043, AVLP594, ...) | 30 | a tone (that data set's prediction, not the literature's) |
| under `curated="all"` only: 151 "unclear" types with another connectome's fast-transmitter prediction | 476 | the sign of their fast synapses |

Tyramine has no place in the model and is ignored. PPL203 is one of the game profile's `class:DAN`
neurons, whose fast synapses the game mutes by hand when the parts list is off; with the parts list on
it keeps the GABA synapses the literature gives it. What the default policy does *not* do is flip the
sign of a confident fast prediction: the literature classes disagree with MaleCNS on the fast
transmitter of 25 types (5,892 neurons) (T3, L3, Mi2, Mi10, Tm39, ...), and where two data sets disagree on a
fast transmitter the model has no way to pick. `curated="all"` lets the literature win there too (the
signs flip, and the three modulatory types above lose their predicted fast synapses) for anyone who
wants to see what that does (`fly-brain --curated all`); the popover shows the disagreement either
way. Classes that are another connectome's type only fill "unclear" predictions, and only under
`all` (the survival table below shows why), and coarse matches (a `_a` type read as its stem's
class, or a class read off one VFB individual) are never used for the model at all.

**Receptor signs.** The one net sign per modulator was the gap the parts list admitted to. VFB
carries, for 276 cell classes, the single-cell RNA-seq clusters of the Fly Cell Atlas
(Li et al. 2022), Davie et al. (2018), the Aging Fly Cell Atlas, Özel et al. (2021) and others,
and for each cluster the fraction of its cells expressing each gene (values of 20 % and above). A
harvest of the 17 aminergic receptor genes (`data/vfb_receptors.json.gz`, 2,117 adult clusters,
CC-BY 4.0) lets the tone's sign follow the target: for a target whose type (or a parent class within
two steps, provided it groups at most 150 kit types) has an adult cluster, the tone's weight is
`clip(Σ sign_r · extent_r, -1, 1)` over that modulator's receptors, +1 for the Gs- and Gq-coupled
ones (Dop1R1, Dop1R2, DopEcR; Oamb, Octβ1R-3R; 5-HT2A, 5-HT2B, 5-HT7) and −1 for the Gi-coupled
ones (Dop2R; Octα2R; 5-HT1A, 5-HT1B; the couplings and their references are in `parts.RECEPTORS`),
so `gain = 1 + Σ_k a_k · s_k · level_k / (level_k + 5)`, floored at 0.1. A receptor a cluster does
not list (under 20 % of its cells) counts as zero, and several clusters of one data set are averaged.
When a class has clusters in several data sets the kit takes one: the Fly Cell Atlas first, then
Davie et al., Özel et al. (adult optic lobe), Baker and Mokashi et al., the Aging Fly Cell Atlas and
the nerve-cord atlases; within one, head or brain before other tissues, male before mixed before
female. Examples: the γ Kenyon cells inherit the adult γ Kenyon cell cluster of Davie et al. 2018
(Dop2R 95 %, Dop1R2 73 %, Dop1R1 70 %, DopEcR 67 %: net dopamine weight clipped at +1; 5-HT1A 48 %,
so serotonin *lowers* their gain, −0.84); the MBONs inherit the adult MBON cluster (Dop2R 79 %,
DopEcR 65 %: dopamine weight −0.14); LC10a's own adult optic-lobe cluster (Özel et al.) has Dop2R
95 %, so dopamine and serotonin both lower its gain; the photoreceptors express only DopEcR among
these receptors. Pupal, larval and embryonic clusters are never used, nor the day-70 aging atlas
(VFB gives it no stage); where no adult cluster exists within two steps up the class tree (most of
the central brain's small types, the giant fibre, MN9, the HS cells) the one-sign rule stands.
Coverage on this connectome: 15,232 of the 38,998 modulated targets have an adult cluster, and
their weight is net inhibitory on 2,004 of them for dopamine, 124 for octopamine and 8,478 for
serotonin.

**What it does to the validated experiments** (game profile; `fly-brain --profile game --parts
--curated off|modulators|all`):

| | parts off | v2.4 parts list | + receptor signs | + curated transmitters (the default) | `curated="all"` |
|---|---|---|---|---|---|
| neurons whose transmitter role changes | 0 | 0 | 0 | 1,301 | 7,699 |
| sugar → MN9 (30-90 Hz) | 50 | 73 | 63 | 50 | **22** |
| vinegar → MBON11 (1-60 Hz) | 1.0 | **0** | 1.0 | **0** | **0** |
| courtship → DNp13 (5-90 Hz) | 39 | 5.0 | 6.2 | 8.8 | 8.8 |
| motion right → DNa02 left (0-20 Hz) | 0 | 20 | **23** | 3.3 | 17 |
| bitter → PPL101 (30-150 Hz) | 78 | 113 | 109 | 136 | 44 |
| fruitless silenced → ps1 song motor neurons (0-3 Hz) | 0 | 2.5 | 0 | 0 | **5.0** |
| **passed** | **16 / 16** | **15 / 16** | **15 / 16** | **15 / 16** | **13 / 16** |

The receptor signs alone bring MBON11 back to 1.0 Hz, inside its range, and move the optomotor
experiment's contralateral DNa02 from 20 to 23 Hz, just past its ceiling. The curated transmitters of the default policy pull that readout back to 3 Hz and
leave one miss, the same marginal MBON11 readout the v2.4 parts list missed (0 Hz against a range
whose lower edge the standard model sits on; v2.6 fixes it, section 8.2). Every other readout stays inside its range, most of
them closer to the standard model's values than with the v2.4 parts list.

`curated="all"` shows why the default takes its fills from the literature only. With the
connectome-derived classes also filling "unclear" predictions, the sugar reflex falls to 22 Hz, and
one type accounts for all of it: GNG578, two neurons MaleCNS calls "unclear" and FlyWire's matching
type CB0087 predicts to be GABAergic, feeds DNge080, a direct excitatory input to MN9. The model's
convention that "unclear" neurons excite had been lending the reflex part of its drive; which of the
two predictions is right is not something either data set can settle. The literature sign flips of
`all` (T3, L3, Mi2, ...) cost the punishment signal half its rate and let a little song leak past
the silenced fruitless neurons. A busy second of brain time costs about the same as with the v2.4
parts list (1.1-1.2 s against 0.8 s with the parts off, compiled backend).

### 8.2 Where APL releases, and the vinegar readout (v2.6)

Every parts-list version up to v2.5 missed one readout: MBON11 (MBON-γ1pedc>α/β) under vinegar,
range 1-60 Hz. The standard model is marginal on it as well. On five seeds, each run on a fresh
brain, MBON11 fires 1, 0, 3, 3, 1 Hz with the parts list off, 0, 2, 0, 1, 0 Hz with the v2.5
default, and 0 Hz on all five with the v2.4 parts list. The real neuron answers odours robustly in
whole-cell recordings (Hige et al. 2015), so the model, not the range, was wrong.

**The cause.** MBON11 has 38,852 synapses from Kenyon cells and 441 from APL, the mushroom body's
GABAergic feedback neuron. APL is the one input that matters. In every configuration up to v2.5 it
fires at 175-190 Hz during vinegar, close to the rate its refractory period allows, so its 441
synapses deliver about as much charge as the 38,852 Kenyon-cell synapses. Over the 500 ms window
APL brings −6,800 to −8,100 mV·spikes against +5,300 to +5,900 from the Kenyon cells. With APL
silenced, MBON11 fires 195 Hz (parts off) or 69 Hz (v2.5 default). The γ Kenyon cells, which give
MBON11 71 % of its Kenyon-cell input, are among the most inhibited: 59 APL synapses per γ-m cell
against 26-46 for the α/β subtypes, and only 3 % of them fire to vinegar. The parts list made it worse. APL has
no cluster in the single-cell atlases (it is one cell per side), so the one-sign rule applied, and
dopamine, octopamine and serotonin all raised its input gain, to about 2x during vinegar.

Two findings about the real APL bear on this:

* **APL does not spike, and its activity stays local.** Its activity and its inhibition of Kenyon
  cells are confined to the part of the mushroom body that is active, so it can inhibit one
  compartment and not another (Amin et al. 2020). The model's APL is one spiking point: each spike
  arrives at all of its 207,598 output synapses at once, in the quiet γ lobe as much as in the busy
  α/β lobes.
* **Dopamine suppresses APL through Dop2R.** Dopamine neurons synapse onto APL and inhibit it
  through the D2-like receptor; knocking Dop2R down in APL impairs aversive learning (Zhou et al.
  2019).

**What was tried first**, on the real connectome (game profile, default parts list):

| change | MBON11 on five fresh seeds (Hz) | Kenyon cells (Hz) | why it is not the answer |
|---|---|---|---|
| APL as a graded cell (the optic-lobe rule) | 0 (seed 0) | 0.8 | APL's input is so large that its voltage sits at 50 mV and its release saturates at 300 events/s: more inhibition, not less |
| APL's receptor fact only (Dop2R) | 13, 1, 0, 0, 0 | 2.0-2.7 | right direction (APL's gain 2.0 → 1.5, rate 190 → 151 Hz), not enough |
| APL's release onto the γ lobe halved, by hand | 33-55 | 2.4-4.2 | shows where the problem is, but the factor is a free parameter |

**What v2.6 does.** Two new tables in `parts.py`, both in the default parts list:

* `RECEPTOR_FACTS`: receptors a cell type is shown to use by direct evidence in that type, for types
  that no atlas cluster covers, and only for the modulators they name. It has one row: APL uses
  Dop2R, so dopamine lowers its gain. Octopamine and serotonin keep the one-sign rule on APL because
  nothing is known.
* `LOCAL`: wide-field neurons whose release follows the activity around each target. It has one row:
  APL, with the three Kenyon-cell lobe systems (γ, α/β, α'/β') as its compartments. Every 10 ms each
  compartment's activity (the Kenyon-cell spikes arriving at APL's synapses in that lobe, decaying
  with the 20 ms membrane time constant) is divided by the number of input synapses it has there.
  Each compartment's density is compared with APL's mean over all its input. APL's release onto a
  target is scaled by `sum_c mix_c · min(1, density_c / mean)`, where `mix` is the target's place:
  a Kenyon cell sits in its own lobe, any other target in proportion to the Kenyon-cell input it
  gets from each lobe (MBON11: 72 % γ, 28 % α/β), and a target with no Kenyon-cell input (a
  projection neuron in the calyx) keeps the whole cell's release. APL keeps its spikes as the
  measure of its overall depolarisation. The cap at 1 is the saturation argument: APL already fires
  at its maximum, so a busy compartment cannot release more than the whole cell does, and a quiet one
  releases less. Nothing in the rule is fitted: the compartments come from the cell types, the mix
  from the synapse counts, and the time constant is the membrane's. At 10 or 50 ms instead of 20,
  MBON11 is 23-39 or 35-47 Hz.

**Results** (game profile; the survival table runs the experiments in sequence on one brain, as
before; the per-seed columns use a fresh brain for each seed, because the mushroom-body learning
rule carries depression from one run to the next):

| configuration | passed | MBON11, five seeds (Hz) | Kenyon cells (Hz) | Kenyon cells active | busy second (s) |
|---|---|---|---|---|---|
| parts off | 16 / 16 | 1, 0, 3, 3, 1 | 1.9-2.1 | 7.8 % (seed 0) | 0.76 |
| v2.4 parts list | 15 / 16 | 0, 0, 0, 0, 0 | 2.0-2.1 | | 1.24 |
| v2.5 default | 15 / 16 | 0, 2, 0, 1, 0 | 1.9-2.0 | 7.5-8.7 % | 1.15 |
| v2.5 + APL's receptor fact | 16 / 16 | 13, 1, 0, 0, 0 | 2.0-2.7 | | 1.15 |
| v2.5 + local APL | 16 / 16 | 38, 21, 29, 10, 32 | 2.1-3.7 | | 1.21 |
| **v2.6 default (both)** | **16 / 16** | **51, 34, 43, 38, 38** | **3.4-4.0** | **12-17 %** | **1.28** |
| v2.6, `curated="off"` | 15 / 16 (DNa02 left 23 Hz, as in v2.5) | 20, 20, 25, 30, 31 | 2.3-2.5 | | 1.24 |
| v2.6, `curated="all"` | 14 / 16 (sugar, fruitless song, as in v2.5) | 30, 25, 17, 22, 22 | 2.2-2.9 | | 1.23 |
| v2.6, `receptor_signs=False` | 15 / 16 (**vinegar overshoots**) | 88, 49, 104, 32, 89 | 2.7-5.6 | | 1.28 |

With v2.6, MBON11 answers vinegar on every seed and every validated experiment passes. During
vinegar APL's release onto the γ lobe runs at 44-71 % of the whole cell's, onto α'/β' at 60-100 %
and onto α/β at 100 %.

**The price, and how it works.** The fix does not work mainly through the γ lobe. APL and DPM
inhibit each other: APL→DPM 1,595 synapses, DPM→APL 3,521 (DPM is GABAergic as well as
serotonergic in the literature, which the default curated policy follows). In v2.5, APL wins and
DPM is silent during vinegar. With local release, APL's grip on DPM loosens (DPM's Kenyon-cell
input is 42-44 % γ, so APL's release onto it drops to about three quarters), DPM wins (164 Hz),
and APL falls to 83 Hz:

| during vinegar, seed 0 | APL (Hz) | DPM (Hz) | Kenyon cells active | MBON11 (Hz) |
|---|---|---|---|---|
| v2.5 default | 190 | 0 | 7.5 % | 0 |
| v2.5 + APL's receptor fact | 151 | 41 | 12.0 % | 13 |
| v2.6 default | 83 | 164 | 16.6 % | 51 |
| v2.6, local release onto Kenyon cells only | 201 | 0 | 8.6 % | 3 |
| v2.6, local release onto everything else only | 19 | 229 | 17.2 % | 65 |

So the Kenyon-cell code is about twice as dense as before: 16.6 % of Kenyon cells fire to vinegar
against 7.5 %, where real flies use roughly 5-10 % (section 4.3). Every subtype is recruited more:
γ-m from 2.9 % to 11.3 %, a step towards the real fly's γ lobe, but α'/β'-m reaches 46 %. With
`curated="off"`, where DPM has no fast GABA, local release still fixes MBON11 (20-31 Hz) and the
Kenyon cells stay at 2.3-2.5 Hz. The switch between APL and DPM is itself an artefact of two spiking
point neurons inhibiting each other. The real pair is also coupled by heterotypic gap junctions
(Wu et al. 2011), which the model does not have. One more limit: without receptor signs every tone
raises the gain, which on top of local release drives the vinegar readouts past their ceilings.
`fly-brain --parts --global-apl` (or `PartsList(local=())`) gives the v2.5 behaviour of APL back
with everything else unchanged.

**Where APL's synapses really are (v2.8).** The kit's connectome holds one synapse count per pair of
neurons and no positions, so v2.6 placed each APL synapse in the lobe of the Kenyon cell it touches.
`tools/harvest_neuprint_rois.py` asks neuPrint (`male-cns:v1.0`, through the repository's GitHub
Actions workflow and its token) for every connection into and out of APL and DPM, split by neuPrint's
primary regions: `data/mb_roi_connectivity.json.gz`, 5,646 neurons and 75,475 pair-region rows. Its
per-pair totals match the kit's synapse counts exactly, and they match the public per-synapse release
of the same data (gs://flyem-male-cns) region by region. About a third of APL's Kenyon-cell synapses
are not where v2.6 put them. 15.5 % are in the calyx and 8 % in the pedunculus, where Kenyon cells of
every lobe meet, and 11 % lie outside every mushroom-body region (the accessory calyces among them).

When the table is present (it ships with the kit), `LOCAL`'s compartments are neuPrint's
mushroom-body regions: calyx, pedunculus, and the γ, α, β, α′ and β′ lobes, per side. A synapse outside
them keeps the whole cell's release. The rule itself is unchanged: each compartment's Kenyon-cell
input density against APL's mean, capped at 1, with the 20 ms time constant. A target's mix is now
where APL's synapses onto it are, so no Kenyon-cell lobe is assumed. Without the table, and on grown or
rewired flies whose pairs the table does not cover, the v2.6 lobe placement is used. During vinegar
(game profile, default parts list), the release onto each region, weighted by APL's output synapses
there, is:

| seed | γ lobe | α′ lobe | β′ lobe | α, β lobes, calyx, pedunculus |
|---|---|---|---|---|
| 0 | 0.57 | 0.92 | 0.87 | 1.0 |
| 1 | 0.53 | 0.88 | 0.86 | 1.0 |

MBON11 fires 43, 44, 36, 33 and 38 Hz on five fresh seeds (v2.6: 51, 34, 43, 38, 38). 16.3-16.4 % of
Kenyon cells fire to vinegar. All 16 validated experiments pass, with the same two fragile readouts as
before (section 2.3). Only vinegar drives the Kenyon cells, so the other 15 give identical numbers. The
change corrects the anatomy but no verdict. It does not make the Kenyon-cell code sparser either, since
the density comes from the APL/DPM switch above, and the switch is unchanged. A separate check with
every unlabelled synapse given the region of its nearest labelled neighbour, and with seven regions
not split by side, gave the same picture on twenty seeds (MBON11 41.0 ± 10.4 Hz, Kenyon cells 15.3 %
active).

### 8.3 Electrical synapses between APL and DPM: tried, not adopted (v2.7)

Section 8.2 left one side effect: with local APL release, DPM wins its tug-of-war with APL during
an odour and about 16 % of Kenyon cells respond instead of 8 %. APL and DPM are also joined by
heterotypic gap junctions (Wu et al. 2011), which the model lacks, and coupling pulls two cells'
voltages together, so the obvious test was to add them.

The prototype coupled each APL to the DPM on its side. Voltage is shared every step at a strength
set by the coupling coefficient CC (the fraction of one cell's steady depolarisation that reaches
the other). As an option, each spike also passes CC x 70 mV to the partner as a spikelet, because
an integrate-and-fire cell resets at a spike and voltage sharing alone never carries the spike
itself. No coupling coefficient has been measured for this pair, so four were tried. Vinegar, game
profile, default parts list, five seeds (APL, DPM and the active fraction are seed 0):

| CC | spikelets | APL (Hz) | DPM (Hz) | Kenyon cells active | Kenyon cells (Hz) | MBON11 (Hz) |
|---|---|---|---|---|---|---|
| none | | 83 | 164 | 16.6 % | 3.4-4.0 | 34-51 |
| 0.05 | no | 96 | 145 | 16.3 % | 3.1-4.2 | 33-55 |
| 0.05 | yes | 42 | 192 | 16.0 % | 3.9-4.3 | 50-59 |
| 0.1 | no | 151 | 65 | 12.2 % | 2.9-4.5 | 30-62 |
| 0.1 | yes | 70 | 149 | 14.7 % | 3.7-3.8 | 46-52 |
| 0.2 | no | 72 | 164 | 16.3 % | 3.5-4.0 | 46-52 |
| 0.2 | yes | 79 | 135 | 13.7 % | 3.1-3.5 | 40-46 |
| 0.3 | no | 59 | 170 | 17.0 % | 3.3-4.7 | 41-66 |
| 0.3 | yes | 97 | 96 | 12.0 % | 3.1-3.2 | 40-45 |

Coupling moves the balance between APL and DPM, but not in any consistent direction. The code never
gets back near 8 %, and at 0.1 and 0.3 without spikelets MBON11 crosses its 60 Hz ceiling on some
seeds. A mechanism with an unmeasured strength that does not do what it was added for would only
add a knob, so the kit does not include it. The electrical synapses of the giant-fibre system
(Phelan et al. 2008) and of the antennal lobe (Yaksi & Wilson 2010) are left out for the same
reason: the giant fibre's chemical synapses already drive TTMn in every validated experiment, and
the antennal-lobe local neurons are silenced in the game profile. The dense code more likely comes
from DPM itself. Its fast GABA onto APL (3,521 synapses) follows the curated transmitter, but DPM is
a spiking point here and has no resting activity, which a real DPM does.

## 9. The female fly (v2.8)

The published model (Shiu et al. 2024) ran on FlyWire, the whole brain of an adult **female** fly
(Dorkenwald et al. 2024; annotations Schlegel et al. 2024). `load_connectome(female=True)`,
`python fly_brain.py --female` and `python fly_game.py --female` run the whole kit on it.

### 9.1 What the female fly is

`virtual_fly/flywire.py` builds it on first use from two public files, each pinned to one commit and
checked by SHA-256: the connectivity table the published model itself uses (`Connectivity_783.parquet`,
philshiu/Drosophila_brain_model) and FlyWire's neuron annotations (`Supplemental_file1_neuron_annotations.tsv`,
flyconnectome/flywire_annotations). Neither is redistributed; the built file is about 45 MB. Reading
the parquet file needs `pyarrow` (`pip install -e ".[female]"`).

* **The published model as published.** The female file keeps every connection (15,091,983 pairs,
  54,492,922 synapses, 139,262 neurons), because the published model uses all of them and its
  0.275 mV per synapse was fitted on them. The kit's male file keeps only connections of 5 or more
  synapses, and its gain of 0.65 is a calibration for the male data (section 1.3). So the default
  gain follows the connectome: 0.65 for the male, **1.0** (the paper's value) for the female
  (`brain.DEFAULT_GAIN`). `build_female(min_synapses=5)` builds a file cut like the male one.
* **Signs** come from each neuron's predicted transmitter in the annotations, the same rule as for
  the male. They agree with the published model's own `Excitatory` column on 98.9 % of the
  connections of 5 or more synapses (the annotations' predictions are newer).
* **Classes** are translated into the male data's vocabulary (e.g. FlyWire's `central` →
  `cb_intrinsic`), so `class:Kenyon_Cell`, `class:ALLN`, `class:DAN` and the profiles work unchanged.
  fru/dsx labels are FlyWire's `fru`, `dsx` and `coexpress` (no confidence grade), and
  `dimorphism:female` selects the 363 female-specific or potentially female-specific neurons.
* **Missing:** FlyWire has no ventral nerve cord, so there are no leg, wing or neck motor neurons (TTMn,
  ps1, hg, DLMn), and male-specific cells such as pIP10 do not exist. The annotations have no medulla
  column coordinates, so the computed column-by-column motion vision (section 5.3) switches itself
  off, and the hand-built feature detectors still drive LC4, LPLC2 and LC10a.

### 9.2 The kit's names in FlyWire

The experiments and senses are written with MaleCNS cell-type names. Where FlyWire calls a cell
something else, `flywire.ALIASES` maps the name, and the table is stored in the female file, where
`Connectome.select` reads it. A real cell type of the same name always wins over an alias.

| kit name | in FlyWire | from |
|---|---|---|
| `MN9` | `CB0701` | FBbt_00111298, whose VFB synonyms include both names; it is the published model's MN9 cell |
| `GNG232` (G2N-1) | `CB0616` | FBbt_00051850 (VFB synonyms G2N-1, GNG232, CB0616); it is the published model's G2N-1 |
| `GNG087` | `CB0219` | FBbt_20004033 (VFB synonyms GNG087, CB0219) |
| `LB3b`, `LB3c` (sugar) | the 20 sugar cells of the published model | FlyWire types all 122 labellar sugar and water cells as `LB3`; the published model's list is one side |
| `LB1a`, `LB1d` | `LB1a,LB1d` | one FlyWire type |
| `LB2a`, `LB2b` | `LB2a-b` | one FlyWire type |
| `prefix:pC1_` | `prefix:pC1` (pC1a-e, 10 cells) | the doublesex pC1 cluster; the male's 148 `pC1_` cells include the male-specific P1 |

### 9.3 Experiments on a fly without some of their neurons

`rate()` of a population that does not exist is 0 Hz, which would pass "MN9 stays silent" without
testing anything. So `run_experiment` checks every population first. A missing stimulus population is
left out (the female has no leg sugar cells, `LgLG3`). A missing readout is reported as n/a and does
not count. The experiment cannot be done (n/a) when its whole stimulus is missing, when a population
it silences is, or when the only readouts left are the stimulated cells themselves. In the female,
three experiments are n/a (the two song-motor ones and the doublesex lesion) and TTMn and pIP10 are
n/a readouts. In the game, gauges for cells the fly lacks are not shown.

### 9.4 What the female fly does

Five seeds each, the male numbers from the same code (parts list off). The ranges are the male fly's
validated results, so on the female they are a comparison, not a test:

| experiment | readout | male | female | range |
|---|---|---|---|---|
| sugar (pure) | G2N-1 | 36.6 | 38.0 | 20-60 |
| sugar (pure) | MN9 | 69.6 | **77.6** | 30-90 |
| sugar (pure) | Fudog | 20.0 | 1.8 | 10-40 |
| bitter (pure) | Scapula / MN9 | 290 / 0 | 130 / **0** | 100-400 / 0-5 |
| sugar + bitter (pure) | MN9 | 0.0 | **3.0** | 0-5 |
| looming (pure) | giant fibre | 342 | 219 | 250-400 |
| dust (pure) | aDN1 / aDN2 | 186 / 136 | 5.6 / 5.8 | 100-260 / 80-200 |
| vinegar (game) | DM1 PN / KC / MBON14 / MBON11 | 289 / 2.0 / 38 / 1.6 | 216 / 2.1 / 57 / 9.8 | all in range |
| bitter → punishment (game) | PPL101 | 82 | 0.0 | 30-150 |
| loud sound (game) | giant fibre | 66 | 7.7 | 20-150 |
| wide-field motion (game) | HS / DNp15 / DNa02 R / DNa02 L | 442 / 180 / 158 / 0 | 348 / 162 / 87 / 44 | ... / 0-20 |
| courtship command (game) | DNp13 | 31.5 | 0.0 | 5-90 |
| sugar, fru silenced (game) | MN9 | 34.5 | 80.8 | 15-60 |

The published model's headline results come out on the female: sugar drives MN9, bitter keeps
it silent, and bitter wins over sugar (pure profile). The antennal grooming route works too with the
published model's own protocol (its 145 Johnston's-organ cells on one side at 220 Hz): its aDN1 fires
at 34 Hz. The kit's "dust" stimulus drives both antennae, and in FlyWire the two sides cancel (one
side alone: aDN1 22 Hz on one side; both: 6 Hz).

Most of the differences are not yet sex differences. The two datasets were reconstructed and their
synapses detected differently: the median neuron has 200 input synapses in FlyWire, all
connections counted, against 344 in the male data, counting only connections of 5 or more synapses.
The giant fibre has 4,300-5,100 input synapses in FlyWire against 19,000-25,000. The sugar cells are
chosen differently (the published model's one-sided list against LB3b and LB3c): they send Fudog 41
synapses, against 195 in the male. The female pC1 cluster is 10 cells against the male's 148, and it
sends DNp13 5 synapses against 1,571. That last one is at least partly biology, since female pC1
lacks the male's P1 cells. Taking a difference between the two flies for biology would need these
confounds removed first. The kit gives both flies, not that answer.

Checked, pure profile: the time step (at 0.1 ms, the paper's Brian2 value: MN9 80, giant fibre 211, aDN1
5.5 Hz), the sign rule (above), and a file cut at 5 synapses like the male's. At gain 0.65 every drive
is weaker there (MN9 46, Scapula 91, giant fibre 167, aDN1 0 Hz); at gain 1.0 it is close to the full
file (giant fibre 208, aDN1 8.8 Hz), except that bitter no longer wins over sugar (MN9 9.8 Hz).

---

## 10. Honest limitations

The starter kit's list, extended. These are the things a neuroscientist would point at first.

1. **Every neuron is identical.** No ion channels, no dendritic geometry, no cell-type-specific
   time constants. Absolute rates mean nothing; only which neurons respond, and roughly how much
   relative to each other, can be trusted. Rates saturate at the refractory limit (500 Hz), which
   the HS cells hit under wide-field motion. The mushroom body's two wide-field neurons, APL and
   DPM, are spiking points that inhibit each other; whichever wins sets the inhibition of the
   whole mushroom body. With the parts list, APL's release follows the Kenyon cells around each
   target, by neuPrint's mushroom-body regions, but its gap junctions with DPM are missing
   (section 8.2; adding them does not make the code sparser, section 8.3).
2. **Runaway loops** are a property of the pure model, not of the fly. The game silences 420
   antennal-lobe local neurons and 340 dopamine neurons and adds fatigue to keep the brain sane;
   these are documented fixes, not physiology. Short-term depression, the physiological brake,
   starves the calibrated sensory chains (section 3).
3. **The central complex rings** for about a second after a strong odour once the antennal lobe
   is quiet (section 3.5), and it is the next circuit to seize if noise is added. Heading
   readouts just after a smell are suspect.
4. **No spontaneous activity.** The walking urge, the wander, hunger and thirst are hand-built.
   Background noise was measured and is available, but the window between "nothing" and
   "seizure" is a few hundredths of a millivolt wide and it triples the cost.
5. **Dopamine is a fast excitatory transmitter** in the data's sign convention, and glutamate is
   inhibitory. The game silences the dopamine neurons' synapses to use them as a slow modulator
   only, and the glutamatergic "avoidance" MBONs can never drive a descending neuron by
   themselves (section 4.8). With the parts list on the tones' signs follow receptor expression
   only where an adult scRNA-seq cluster exists (section 8.1); elsewhere one sign per transmitter
   still stands, and where the literature and the connectome disagree on a *fast* transmitter the
   prediction is kept.
6. **Reward is injected.** Sugar does not reach the PAM dopamine neurons through this wiring
   (section 4.5); eating sugar drives them by hand. Punishment does come from the wiring, and the
   shock tool is a labelled injection.
7. **The learning rule's constants are guesses**, the rule covers forward pairing only, and the
   MBON→behaviour link is hand-built (section 4.8): the brain learns, the steering is scripted from
   what it learned.
8. **The Kenyon-cell code is biased.** At the gain that makes the code sparse and specific, the
   α/β and α'/β' Kenyon cells carry the code and the γ cells barely fire (KCg-m 3-4 % active;
   KCg-d, KCab-p, KCa'b'-ap1 never), which is not how real flies store short-term memory. KC
   recruitment also depends strongly on which glomeruli an odour drives (7.9 % for one four-glomerulus
   mixture, 2.1 % for another, under 1 % for single glomeruli); only the vinegar mixture was measured.
9. **Vision is half computed.** T4/T5 motion detection is computed per column and the optomotor
   reflex comes out of the wiring; looming and small-object detection are still hand-built
   feature detectors, because LC4 gets no motion input and LC10a cannot be driven from columnar
   cells at all (section 5). The correlator, the facet-to-column map and the preferred-direction
   axes are calibrated from the data but are approximations (the lattice is treated as flat and
   uniform; a and c axes are 97° apart, not 90°). Photoreceptors and the lamina do nothing.
10. **Courtship is primed by hand.** The pheromone route to pC1 is real but ~8x too weak here, so
    tapping the female also drives pC1 directly (section 6.1). The female the male courts in the game has no brain.
11. **Wind is a whisper**, deliberately: the wind-sensing neurons drive grooming and backing in
    this model and no descending neuron encodes wind direction (section 6.4), so upwind search is
    hand-built.
12. **Descending neurons barely move the leg motor neurons** (a few Hz at most, at any gain,
    section 6.5), so the decoder's weights are chosen by hand; the wiring only says which motor
    pools each DN reaches. The body is a drawing with speeds chosen to look right.
13. **Single seeds.** Most probe numbers are one Poisson realisation (seed 0); where a second or
    third seed was run (KC reliability, MBON baselines, the DNa02 disinhibition, the looming and
    optomotor cases, one depression grid point) the picture held, but rates moved by ±10-20 Hz and
    the pure sugar experiment can flip into a runaway on a different draw. MBON30 is bistable
    across seeds.
14. **One brain of each sex.** Left/right asymmetries (e.g. the DNa02 right-side bias under bilateral MBON
    drive, the right-only wind-responsive DNs) may be features of this individual, its
    reconstruction, or the 5-synapse threshold, not of flies. The female fly is another individual,
    reconstructed differently, so a male-female difference is not yet a sex difference (section 9.4).
15. **Not modelled at all:** hormones, neuropeptides (the ontology names the peptidergic types;
    nothing is done with them), neuromodulation beyond the tones and the two hand-built cases
    (hunger on the sugar neurons, dopamine gating), electrical synapses (the ontology knows the
    giant fibre's, the model has none), development, the real leg controller, and anything about
    consciousness, which is not on the table.

---

## 11. References

* Ache JM, Polsky J, Alghailani S, Parekh R, Breads P, Peek MY, Bock DD, von Reyn CR, Card GM
  (2019). Neural basis for looming size and velocity encoding in the *Drosophila* giant fiber
  escape pathway. *Current Biology* 29(6):1073-1081. doi:10.1016/j.cub.2019.01.079
* Amin H, Apostolopoulou AA, Suárez-Grimalt R, Vrontou E, Lin AC (2020). Localized inhibition in the
  *Drosophila* mushroom body. *eLife* 9:e56954. doi:10.7554/eLife.56954
* Arenz A, Drews MS, Richter FG, Ammer G, Borst A (2017). The temporal tuning of the *Drosophila*
  motion detectors is determined by the dynamics of their input elements. *Current Biology*
  27(7):929-944. doi:10.1016/j.cub.2017.01.051
* Aso Y, Hattori D, Yu Y, Johnston RM, Iyer NA, Ngo T-TB, Dionne H, Abbott LF, Axel R, Tanimoto H,
  Rubin GM (2014a). The neuronal architecture of the mushroom body provides a logic for associative
  learning. *eLife* 3:e04577. doi:10.7554/eLife.04577
* Aso Y, Sitaraman D, Ichinose T, Kaun KR, Vogt K, Belliart-Guérin G, Plaçais P-Y, Robie AA,
  Yamagata N, Schnaitmann C, Rowell WJ, Johnston RM, Ngo T-TB, Chen N, Korff W, Nitabach MN,
  Heberlein U, Preat T, Branson KM, Tanimoto H, Rubin GM (2014b). Mushroom body output neurons
  encode valence and guide memory-based action selection in *Drosophila*. *eLife* 3:e04580.
  doi:10.7554/eLife.04580
* Behnia R, Clark DA, Carter AG, Clandinin TR, Desplan C (2014). Processing properties of ON and OFF
  pathways for *Drosophila* motion detection. *Nature* 512:427-430. doi:10.1038/nature13427
* Berg S, Beckett IR, Costa M, Schlegel P, Januszewski M, et al. (2026). Sexual dimorphism in the
  complete *Drosophila* male central nervous system connectome. *Cell* 189(18):5504-5526.
  doi:10.1016/j.cell.2026.08.015. Data: MaleCNS v1.0, CC BY 4.0, https://male-cns.janelia.org/
* Blenau W, Daniel S, Balfanz S, Thamm M, Baumann A (2017). Dm5-HT2B: pharmacological
  characterization of the fifth serotonin receptor subtype of *Drosophila melanogaster*. *Frontiers
  in Systems Neuroscience* 11:28. doi:10.3389/fnsys.2017.00028
* Blenau W, Thamm M (2011). Distribution of serotonin (5-HT) and its receptors in the insect brain
  with focus on the mushroom bodies. *Arthropod Structure & Development* 40(5):381-394.
  doi:10.1016/j.asd.2011.01.004
* Colas JF, Launay JM, Kellermann O, Rosay P, Maroteaux L (1995). *Drosophila* 5-HT2 serotonin
  receptor: coexpression with fushi-tarazu during segmentation. *PNAS* 92(12):5441-5445.
  doi:10.1073/pnas.92.12.5441
* Costa M, Reeve S, Grumbling G, Osumi-Sutherland D (2013). The Drosophila anatomy ontology.
  *Journal of Biomedical Semantics* 4:32. doi:10.1186/2041-1480-4-32
* Court R, Costa M, Pilgrim C, Millburn G, Holmes A, McLachlan A, Larkin A, Matentzoglu N,
  Kir H, Parkinson H, Brown NH, O'Kane CJ, Armstrong JD, Jefferis GSXE, Osumi-Sutherland D (2023).
  Virtual Fly Brain: an interactive atlas of the *Drosophila* nervous system. *Frontiers in
  Physiology* 14:1076533. doi:10.3389/fphys.2023.1076533
* Cohn R, Morantte I, Ruta V (2015). Coordinated and compartmentalized neuromodulation shapes
  sensory processing in *Drosophila*. *Cell* 163(7):1742-1755. doi:10.1016/j.cell.2015.11.019
* Davie K, Janssens J, Koldere D, De Waegeneer M, Pech U, Kreft Ł, Aibar S, Makhzami S,
  Christiaens V, Bravo González-Blas C, Poovathingal S, Hulselmans G, Spanier KI, Moerman T,
  Vanspauwen B, Geurs S, Voet T, Lammertyn J, Thienpont B, Liu S, Konstantinides N, Fiers M,
  Verstreken P, Aerts S (2018). A single-cell transcriptome atlas of the aging *Drosophila*
  brain. *Cell* 174(4):982-998. doi:10.1016/j.cell.2018.05.057
* Davis FP, Nern A, Picard S, Reiser MB, Rubin GM, Eddy SR, Henry GL (2020). A genetic, genomic,
  and computational resource for exploring neural circuit function. *eLife* 9:e50901.
  doi:10.7554/eLife.50901
* Dacks AM, Green DS, Root CM, Nighorn AJ, Wang JW (2009). Serotonin modulates olfactory processing
  in the antennal lobe of *Drosophila*. *Journal of Neurogenetics* 23(4):366-377.
  doi:10.3109/01677060903085722
* Dorkenwald S, Matsliah A, Sterling AR, Schlegel P, Yu SC, et al. (2024). Neuronal wiring diagram of an
  adult brain. *Nature* 634:124-138. doi:10.1038/s41586-024-07558-y (FlyWire, the female fly)
* El-Kholy S, Stephano F, Li Y, Bhandari A, Fink C, Roeder T (2015). Expression analysis of
  octopamine and tyramine receptors in *Drosophila*. *Cell and Tissue Research* 361(3):669-684.
  doi:10.1007/s00441-015-2137-4
* Farrell JA, Murlis J, Long X, Li W, Cardé RT (2002). Filament-based atmospheric dispersion model
  to achieve short time-scale structure of odor plumes. *Environmental Fluid Mechanics* 2:143-169.
* fly-brain-minecraft (blendi-remade). The compact connectome file, the gain of 0.65, the
  Kenyon-cell input scaling and many of the sensory and motor neuron choices.
  https://github.com/blendi-remade/fly-brain-minecraft (commit `6cfa301`; code MIT, data CC BY 4.0).
* Gruntman E, Romani S, Reiser MB (2018). Simple integration of fast excitation and offset, delayed
  inhibition computes directional selectivity in *Drosophila*. *Nature Neuroscience* 21:250-257.
  doi:10.1038/s41593-017-0046-4
* Haag J, Borst A (1996). Amplification of high-frequency synaptic inputs by active dendritic
  membrane processes. *Nature* 379:639-641. doi:10.1038/379639a0
* Han K-A, Millar NS, Grotewiel MS, Davis RL (1996). DAMB, a novel dopamine receptor expressed
  specifically in *Drosophila* mushroom bodies. *Neuron* 16(6):1127-1135.
  doi:10.1016/S0896-6273(00)80139-7
* Han K-A, Millar NS, Davis RL (1998). A novel octopamine receptor with preferential expression in
  *Drosophila* mushroom bodies. *Journal of Neuroscience* 18(10):3650-3658.
  doi:10.1523/JNEUROSCI.18-10-03650.1998
* Hallem EA, Carlson JR (2006). Coding of odors by a receptor repertoire. *Cell* 125(1):143-160.
  doi:10.1016/j.cell.2006.01.050
* Hampel S, Franconville R, Simpson JH, Seeds AM (2015). A neural command circuit for grooming
  movement control. *eLife* 4:e08758. doi:10.7554/eLife.08758
* Handler A, Graham TGW, Cohn R, Morantte I, Siliciano AF, Zeng J, Li Y, Ruta V (2019). Distinct
  dopamine receptor pathways underlie the temporal sensitivity of associative learning. *Cell*
  178(1):60-75. doi:10.1016/j.cell.2019.05.040
* Hardie RC, Raghu P (2001). Visual transduction in *Drosophila*. *Nature* 413:186-193.
  doi:10.1038/35093002
* Hassenstein B, Reichardt W (1956). Systemtheoretische Analyse der Zeit-, Reihenfolgen- und
  Vorzeichenauswertung bei der Bewegungsperzeption des Rüsselkäfers *Chlorophanus*. *Zeitschrift
  für Naturforschung B* 11:513-524.
* Hearn MG, Ren Y, McGrath EW, Grant HR, Ziedonis DM, Hearn GC (2002). A *Drosophila* dopamine
  2-like receptor: molecular characterization and identification of multiple alternatively spliced
  variants. *PNAS* 99(22):14554-14559. doi:10.1073/pnas.202498299
* Himmelreich S, Masuho I, Berry JA, MacMullen C, Skamangas NK, Martemyanov KA, Davis RL (2017).
  Dopamine receptor DAMB signals via Gq to mediate forgetting in *Drosophila*. *Cell Reports*
  21(8):2074-2081. doi:10.1016/j.celrep.2017.10.108
* Hige T, Aso Y, Modi MN, Rubin GM, Turner GC (2015). Heterosynaptic plasticity underlies aversive
  olfactory learning in *Drosophila*. *Neuron* 88(5):985-998. doi:10.1016/j.neuron.2015.11.003
* Inagaki HK, Ben-Tabou de-Leon S, Wong AM, Jagadish S, Ishimoto H, Barnea G, Kitamoto T, Axel R,
  Anderson DJ (2012). Visualizing neuromodulation in vivo: TANGO-mapping of dopamine signaling
  reveals appetite control of sugar sensing. *Cell* 148(3):583-595.
* Karam CS, Jones SK, Javitch JA (2020). Come Fly with Me: an overview of dopamine receptors in
  *Drosophila melanogaster*. *Basic & Clinical Pharmacology & Toxicology* 126(S6):56-65.
  doi:10.1111/bcpt.13277
* Keleş MF, Frye MA (2017). Object-detecting neurons in *Drosophila*. *Current Biology*
  27(5):762-766. doi:10.1016/j.cub.2017.01.012
* Kim AJ, Fitzgerald JK, Maimon G (2015). Cellular evidence for efference copy in *Drosophila*
  visuomotor processing. *Nature Neuroscience* 18:1247-1255. doi:10.1038/nn.4083
* Klapoetke NC, Nern A, Peek MY, Rogers EM, Breads P, Rubin GM, Reiser MB, Card GM (2017).
  Ultra-selective looming detection from radial motion opponency. *Nature* 551:237-241.
  doi:10.1038/nature24626
* Kurmangaliyev YZ, Yoo J, Valdes-Aleman J, Sanfilippo P, Zipursky SL (2020). Transcriptional
  programs of circuit assembly in the *Drosophila* visual system. *Neuron* 108(6):1045-1057.
  doi:10.1016/j.neuron.2020.10.006
* Laughlin SB, Hardie RC (1978). Common strategies for light adaptation in the peripheral visual
  systems of fly and dragonfly. *Journal of Comparative Physiology A* 128:319-340.
  doi:10.1007/BF00657606
* Li H, Janssens J, De Waegeneer M, Kolluru SS, Davie K, Gardeux V, Saelens W, David FPA,
  Brbić M, Spanier K, Leskovec J, McLaughlin CN, Xie Q, Jones RC, Brueckner K, Shim J, Tattikota
  SG, Schnorrer F, Rust K, Nystul TG, Carvalho-Santos Z, Ribeiro C, Pal S, Mahadevaraju S, Przytycka
  TM, Allen AM, Goodwin SF, Berry CW, Fuller MT, White-Cooper H, Matunis EL, DiNardo S, Galenza A,
  O'Brien LE, Dow JAT, FCA Consortium, Jasper H, Oliver B, Perrimon N, Deplancke B, Quake SR,
  Luo L, Aerts S (2022). Fly Cell Atlas: a single-nucleus transcriptomic atlas of the adult fruit
  fly. *Science* 375(6584):eabk2432. doi:10.1126/science.abk2432
* Maisak MS, Haag J, Ammer G, Serbe E, Meier M, Leonhardt A, Schilling T, Bahl A, Rubin GM, Nern A,
  Dickson BJ, Reiff DF, Hopp E, Borst A (2013). A directional tuning map of *Drosophila* elementary
  motion detectors. *Nature* 500:212-216. doi:10.1038/nature12320
* Maqueira B, Chatwin H, Evans PD (2005). Identification and characterization of a novel family
  of *Drosophila* β-adrenergic-like octopamine G-protein coupled receptors. *Journal of
  Neurochemistry* 94(2):547-560. doi:10.1111/j.1471-4159.2005.03251.x
* Meier M, Borst A (2019). Extreme compartmentalization in a *Drosophila* amacrine cell. *Current
  Biology* 29(9):1545-1550. doi:10.1016/j.cub.2019.03.070
* Münch D, Galizia CG (2016). DoOR 2.0 - comprehensive mapping of *Drosophila melanogaster*
  odorant responses. *Scientific Reports* 6:21841.
* Nagel KI, Wilson RI (2011). Biophysical mechanisms underlying olfactory receptor neuron dynamics.
  *Nature Neuroscience* 14:208-216. doi:10.1038/nn.2725
* Özel MN, Simon F, Jafari S, Holguera I, Chen Y-C, Benhra N, El-Danaf RN, Kapuralin K, Malin
  JA, Konstantinides N, Desplan C (2021). Neuronal diversity and convergence in a visual system
  developmental atlas. *Nature* 589(7840):88-95. doi:10.1038/s41586-020-2879-3
* Phelan P, Goulding LA, Tam JLY, Allen MJ, Dawber RJ, Davies JA, Bacon JP (2008). Molecular mechanism
  of rectification at identified electrical synapses in the *Drosophila* giant fiber system. *Current
  Biology* 18(24):1955-1960. doi:10.1016/j.cub.2008.10.067
* Qi Y-X, Xu G, Gu G-X, Mao F, Ye G-Y, Liu W, Huang J (2017). A new *Drosophila* octopamine
  receptor responds to serotonin. *Insect Biochemistry and Molecular Biology* 90:61-70.
  doi:10.1016/j.ibmb.2017.09.010
* Ribeiro IMA, Drews M, Bahl A, Machacek C, Borst A, Dickson BJ (2018). Visual projection neurons
  mediating directed courtship in *Drosophila*. *Cell* 174(3):607-621. doi:10.1016/j.cell.2018.06.020
* Saudou F, Boschert U, Amlaiky N, Plassat J-L, Hen R (1992). A family of *Drosophila* serotonin
  receptors with distinct intracellular signalling properties and expression patterns. *EMBO
  Journal* 11(1):7-17. doi:10.1002/j.1460-2075.1992.tb05021.x
* Schnell B, Joesch M, Forstner F, Raghu SV, Otsuna H, Ito K, Borst A, Reiff DF (2010). Processing
  of horizontal optic flow in three visual interneurons of the *Drosophila* brain. *Journal of
  Neurophysiology* 103(3):1646-1657. doi:10.1152/jn.00950.2009
* Schlegel P, Yin Y, Bates AS, Dorkenwald S, Eichler K, et al. (2024). Whole-brain annotation and
  multi-connectome cell typing of *Drosophila*. *Nature* 634:139-152. doi:10.1038/s41586-024-07686-5
* Shiu PK, Sterne GR, Spiller N, Blumenthal E, et al. (2024). A *Drosophila* computational brain
  model reveals sensorimotor processing. *Nature* 634:210-219. doi:10.1038/s41586-024-07763-9
* Stensmyr MC, Dweck HKM, Farhan A, Ibba I, Strutz A, Mukunda L, Linz J, Grabe V, Steck K,
  Lavista-Llanos S, Wicher D, Sachse S, Knaden M, Becher PG, Seki Y, Hansson BS (2012). A conserved
  dedicated olfactory circuit for detecting harmful microbes in *Drosophila*. *Cell*
  151(6):1345-1357. doi:10.1016/j.cell.2012.09.046
* Srivastava DP, Yu EJ, Kennedy K, Chatwin H, Reale V, Hamon M, Smith T, Evans PD (2005). Rapid,
  nongenomic responses to ecdysteroids and catecholamines mediated by a novel *Drosophila*
  G-protein-coupled receptor. *Journal of Neuroscience* 25(26):6145-6155.
  doi:10.1523/JNEUROSCI.1005-05.2005
* Sugamori KS, Demchyshyn LL, McConkey F, Forte MA, Niznik HB (1995). A primordial dopamine
  D1-like adenylyl cyclase-linked receptor from *Drosophila melanogaster* displaying poor affinity
  for benzazepines. *FEBS Letters* 362(2):131-138. doi:10.1016/0014-5793(95)00224-W
* Suh GSB, Wong AM, Hergarden AC, Wang JW, Simon AF, Benzer S, Axel R, Anderson DJ (2004). A single
  population of olfactory sensory neurons mediates an innate avoidance behaviour in *Drosophila*.
  *Nature* 431:854-859.
* Suver MP, Mamiya A, Dickinson MH (2012). Octopamine neurons mediate flight-induced modulation of
  visual processing in *Drosophila*. *Current Biology* 22(24):2294-2302.
  doi:10.1016/j.cub.2012.10.034
* Suver MP, Matheson AMM, Sarkar S, Damiata M, Schoppik D, Nagel KI (2019). Encoding of wind
  direction by central neurons in *Drosophila*. *Neuron* 102(4):828-842.
  doi:10.1016/j.neuron.2019.03.012
* Takemura S-y, Nern A, Chklovskii DB, Scheffer LK, Rubin GM, Meinertzhagen IA (2017). The
  comprehensive connectome of a neural substrate for 'ON' motion detection in *Drosophila*. *eLife*
  6:e24394. doi:10.7554/eLife.24394
* Tempel BL, Bonini N, Dawson DR, Quinn WG (1983). Reward learning in normal and mutant
  *Drosophila*. *PNAS* 80(5):1482-1486. doi:10.1073/pnas.80.5.1482
* Tsodyks MV, Markram H (1997). The neural code between neocortical pyramidal neurons depends on
  neurotransmitter release probability. *PNAS* 94(2):719-723. doi:10.1073/pnas.94.2.719
* Tully T, Quinn WG (1985). Classical conditioning and retention in normal and mutant *Drosophila
  melanogaster*. *Journal of Comparative Physiology A* 157:263-277. doi:10.1007/BF01350033
* Witz P, Amlaiky N, Plassat J-L, Maroteaux L, Borrelli E, Hen R (1990). Cloning and
  characterization of a *Drosophila* serotonin receptor that activates adenylate cyclase. *PNAS*
  87(22):8940-8944. doi:10.1073/pnas.87.22.8940
* Wu C-L, Shih M-FM, Lai JS-Y, Yang H-T, Turner GC, Chen L, Chiang A-S (2011). Heterotypic gap
  junctions between two neurons in the *Drosophila* brain are critical for memory. *Current Biology*
  21(10):848-854. doi:10.1016/j.cub.2011.02.041
* Yaksi E, Wilson RI (2010). Electrical coupling between olfactory glomeruli. *Neuron* 67(6):1034-1047.
  doi:10.1016/j.neuron.2010.08.041
* Yang HH, St-Pierre F, Sun X, Ding X, Lin MZ, Clandinin TR (2016). Subcellular imaging of voltage and
  calcium signals reveals neural processing in vivo. *Cell* 166(1):245-257.
  doi:10.1016/j.cell.2016.05.031
* Yorozu S, Wong A, Fischer BJ, Dankert H, Kernan MJ, Kamikouchi A, Ito K, Anderson DJ (2009).
  Distinct sensory representations of wind and near-field sound in the *Drosophila* brain.
  *Nature* 458:201-205.
* Zheng L, de Polavieja GG, Wolfram V, Asyali MH, Hardie RC, Juusola M (2006). Feedback network
  controls photoreceptor output at the layer of first visual synapses in *Drosophila*. *Journal of
  General Physiology* 127(5):495-510. doi:10.1085/jgp.200509470
* Zhou M, Chen N, Tian J, Zeng J, Zhang Y, Zhang X, Guo J, Sun J, Li Y, Guo A, Li Y (2019). Suppression
  of GABAergic neurons through D2-like receptor secures efficient conditioning in *Drosophila*
  aversive olfactory learning. *PNAS* 116(11):5118-5125. doi:10.1073/pnas.1812342116
