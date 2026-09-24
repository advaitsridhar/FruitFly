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

* Every 20 steps, `v` and `g` values below 1 µV (`FLUSH_MV`, 7,000x below threshold) are snapped
  to 0. Without this, values decaying for hundreds of milliseconds drift into the float32 denormal
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
* head bristles `BM_InOm` → `MDN` (backing away from a bump).
* `pC1` courtship neurons → `pIP10` (song).

---

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
site of short-term memory, and it biases which compartments can learn (section 7).

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

**Preferred directions** (`step1c`): the dendrite of a T4 cell samples Mi9 on its leading side
and Mi4/C3 on its trailing side (Takemura et al. 2017), so the offset between the centroids of a
T4's Mi9 inputs and its Mi4 inputs points along its null-to-preferred axis. Per subtype and side
(hex units; standard error ≤ 0.03; 440-764 cells each):

| subtype | Mi9 − Mi4, left | Mi9 − Mi4, right | angle in the 60° hex plane (L / R) | direction |
|---|---|---|---|---|
| T4a | (+0.85, −1.17) | (+0.91, −1.16) | −75° / −72° | front-to-back |
| T4b | (−0.88, +1.28) | (−0.84, +1.34) | +102° / +98° | back-to-front |
| T4c | (−1.74, −1.35) | (−1.79, −1.42) | −154° / −154° | up |
| T4d | (+1.71, +1.90) | (+1.74, +1.88) | +32° / +31° | down |

a/b are antiparallel along `hex2 − hex1`, c/d along `hex1 + hex2`, and the T5 offsets
(Tm9 − Tm1) point the same way. The two sides agree within 3°. `COLUMNAR_AXES` in
`senses/vision.py` uses a = −74°, c = −154° in the 60°-axis plane
(`cx = hex1 + 0.5·hex2, cy = 0.866·hex2`), projects every column onto these two axes and spreads
the result linearly over the eye's field (azimuth −8° to 165°, elevation −55° to 60°). Distances
along the lattice are not exact angles, but every column lands on the right part of the visual
field and every subtype points the right way, which is what the downstream wiring needs. In this
frame the a and c axes are 97° apart rather than 90°; the real lattice is tilted relative to the
equator.

Two independent checks that the frame is right:

* **T4/T5 → lobula-plate tangential cells** (right side, synapses): T4a → HSE 6951, HSN 6526,
  HSS 6741 (and LPi12 28,678); T4b → H2 12,235 (HS 0); T4c → LPi34 39,231 (VS 6); T4d → VS 24,437;
  T5a → HSE 8317, HSN 8219, HSS 7037; T5b → H2 14,558; T5d → VS 24,463. So a/b are the
  horizontal system (layers 1/2) and c/d the vertical system (layers 3/4), as in Maisak et al.
  2013.
* **LPLC2 receptive fields**: for each of the 91 right LPLC2 cells the mean visual offset of its
  T4/T5 inputs from its own centre is a: +2.96 columns azimuth (behind), b: −2.39 (in front),
  c: +2.64 elevation (above), d: −2.82 (below), standard error ~0.1. That is the outward-motion
  arrangement Klapoetke et al. 2017 measured physiologically, recovered from wiring alone.

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

## 7. Honest limitations

The starter kit's list, extended. These are the things a neuroscientist would point at first.

1. **Every neuron is identical.** No ion channels, no dendritic geometry, no cell-type-specific
   time constants. Absolute rates mean nothing; only which neurons respond, and roughly how much
   relative to each other, can be trusted. Rates saturate at the refractory limit (500 Hz), which
   the HS cells hit under wide-field motion.
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
   themselves (section 4.8).
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
    tapping the female also drives pC1 directly (section 6.1). The female has no brain.
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
14. **One brain.** Left/right asymmetries (e.g. the DNa02 right-side bias under bilateral MBON
    drive, the right-only wind-responsive DNs) may be features of this individual, its
    reconstruction, or the 5-synapse threshold, not of flies.
15. **Not modelled at all:** hormones, neuromodulation beyond the two hand-built cases (hunger on
    the sugar neurons, dopamine gating), electrical synapses, development, the real leg controller,
    and anything about consciousness, which is not on the table.

---

## 8. References

* Aso Y, Hattori D, Yu Y, Johnston RM, Iyer NA, Ngo T-TB, Dionne H, Abbott LF, Axel R, Tanimoto H,
  Rubin GM (2014a). The neuronal architecture of the mushroom body provides a logic for associative
  learning. *eLife* 3:e04577. doi:10.7554/eLife.04577
* Aso Y, Sitaraman D, Ichinose T, Kaun KR, Vogt K, Belliart-Guérin G, Plaçais P-Y, Robie AA,
  Yamagata N, Schnaitmann C, Rowell WJ, Johnston RM, Ngo T-TB, Chen N, Korff W, Nitabach MN,
  Heberlein U, Preat T, Branson KM, Tanimoto H, Rubin GM (2014b). Mushroom body output neurons
  encode valence and guide memory-based action selection in *Drosophila*. *eLife* 3:e04580.
  doi:10.7554/eLife.04580
* Ache JM, Polsky J, Alghailani S, Parekh R, Breads P, Peek MY, Bock DD, von Reyn CR, Card GM
  (2019). Neural basis for looming size and velocity encoding in the *Drosophila* giant fiber
  escape pathway. *Current Biology* 29(6):1073-1081. doi:10.1016/j.cub.2019.01.079
* Berg S, Beckett IR, Costa M, Schlegel P, Januszewski M, et al. (2026). Sexual dimorphism in the
  complete *Drosophila* male central nervous system connectome. *Cell* 189(18):5504-5526.
  doi:10.1016/j.cell.2026.08.015. Data: MaleCNS v1.0, CC BY 4.0, https://male-cns.janelia.org/
* Farrell JA, Murlis J, Long X, Li W, Cardé RT (2002). Filament-based atmospheric dispersion model
  to achieve short time-scale structure of odor plumes. *Environmental Fluid Mechanics* 2:143-169.
* Hallem EA, Carlson JR (2006). Coding of odors by a receptor repertoire. *Cell* 125(1):143-160.
  doi:10.1016/j.cell.2006.01.050
* Hampel S, Franconville R, Simpson JH, Seeds AM (2015). A neural command circuit for grooming
  movement control. *eLife* 4:e08758. doi:10.7554/eLife.08758
* Handler A, Graham TGW, Cohn R, Morantte I, Siliciano AF, Zeng J, Li Y, Ruta V (2019). Distinct
  dopamine receptor pathways underlie the temporal sensitivity of associative learning. *Cell*
  178(1):60-75. doi:10.1016/j.cell.2019.05.040
* Hassenstein B, Reichardt W (1956). Systemtheoretische Analyse der Zeit-, Reihenfolgen- und
  Vorzeichenauswertung bei der Bewegungsperzeption des Rüsselkäfers *Chlorophanus*. *Zeitschrift
  für Naturforschung B* 11:513-524.
* Hige T, Aso Y, Modi MN, Rubin GM, Turner GC (2015). Heterosynaptic plasticity underlies aversive
  olfactory learning in *Drosophila*. *Neuron* 88(5):985-998. doi:10.1016/j.neuron.2015.11.003
* Inagaki HK, Ben-Tabou de-Leon S, Wong AM, Jagadish S, Ishimoto H, Barnea G, Kitamoto T, Axel R,
  Anderson DJ (2012). Visualizing neuromodulation in vivo: TANGO-mapping of dopamine signaling
  reveals appetite control of sugar sensing. *Cell* 148(3):583-595.
* Keleş MF, Frye MA (2017). Object-detecting neurons in *Drosophila*. *Current Biology*
  27(5):762-766. doi:10.1016/j.cub.2017.01.012
* Kim AJ, Fitzgerald JK, Maimon G (2015). Cellular evidence for efference copy in *Drosophila*
  visuomotor processing. *Nature Neuroscience* 18:1247-1255. doi:10.1038/nn.4083
* Klapoetke NC, Nern A, Peek MY, Rogers EM, Breads P, Rubin GM, Reiser MB, Card GM (2017).
  Ultra-selective looming detection from radial motion opponency. *Nature* 551:237-241.
  doi:10.1038/nature24626
* Maisak MS, Haag J, Ammer G, Serbe E, Meier M, Leonhardt A, Schilling T, Bahl A, Rubin GM, Nern A,
  Dickson BJ, Reiff DF, Hopp E, Borst A (2013). A directional tuning map of *Drosophila* elementary
  motion detectors. *Nature* 500:212-216. doi:10.1038/nature12320
* Münch D, Galizia CG (2016). DoOR 2.0 - comprehensive mapping of *Drosophila melanogaster*
  odorant responses. *Scientific Reports* 6:21841.
* Nagel KI, Wilson RI (2011). Biophysical mechanisms underlying olfactory receptor neuron dynamics.
  *Nature Neuroscience* 14:208-216. doi:10.1038/nn.2725
* Ribeiro IMA, Drews M, Bahl A, Machacek C, Borst A, Dickson BJ (2018). Visual projection neurons
  mediating directed courtship in *Drosophila*. *Cell* 174(3):607-621. doi:10.1016/j.cell.2018.06.020
* Shiu PK, Sterne GR, Spiller N, Blumenthal E, et al. (2024). A *Drosophila* computational brain
  model reveals sensorimotor processing. *Nature* 634:210-219. doi:10.1038/s41586-024-07763-9
* Stensmyr MC, Dweck HKM, Farhan A, Ibba I, Strutz A, Mukunda L, Linz J, Grabe V, Steck K,
  Lavista-Llanos S, Wicher D, Sachse S, Knaden M, Becher PG, Seki Y, Hansson BS (2012). A conserved
  dedicated olfactory circuit for detecting harmful microbes in *Drosophila*. *Cell*
  151(6):1345-1357. doi:10.1016/j.cell.2012.09.046
* Suh GSB, Wong AM, Hergarden AC, Wang JW, Simon AF, Benzer S, Axel R, Anderson DJ (2004). A single
  population of olfactory sensory neurons mediates an innate avoidance behaviour in *Drosophila*.
  *Nature* 431:854-859.
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
* Yorozu S, Wong A, Fischer BJ, Dankert H, Kernan MJ, Kamikouchi A, Ito K, Anderson DJ (2009).
  Distinct sensory representations of wind and near-field sound in the *Drosophila* brain.
  *Nature* 458:201-205.
* fly-brain-minecraft (blendi-remade). The compact connectome file, the gain of 0.65, the
  Kenyon-cell input scaling and many of the sensory and motor neuron choices.
  https://github.com/blendi-remade/fly-brain-minecraft (commit `6cfa301`; code MIT, data CC BY 4.0).
