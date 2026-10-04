# Two flies: progress log

Plan: docs/TWO_FLIES_PLAN.md. Newest session notes first.

## Status
| phase | branch | PR | state | last update |
|---|---|---|---|---|
| 0 | claude/two-flies-p0-baseline | #16 | merged (squash, `3631770`) | 2026-09-27 |
| 1 | claude/two-flies-p1-two-brains | #17 | merged (squash, `f70d8be`) | 2026-10-01 |
| 2 | claude/two-flies-p2-gpu | #18 | merged (squash, `63681db`) | 2026-10-03 |
| 3 | claude/two-flies-p3-3d-view | #19 | merged (squash, `157ffa9`) | 2026-10-03 |
| 4 | claude/two-flies-p4-physics-pair | #20 | merged (squash, `5a27efe`) | 2026-10-04 |
| 5 | claude/two-flies-p5-banc | #21 | merged (squash, `ff16d8c`) | 2026-10-04 |
| real time | claude/flygym2-realtime | draft | the flygym 2.1 migration, v3.0.0: built, measured, a draft PR; the owner's word needed on the physics golden hashes | 2026-10-04 |

## This machine
- OS: WSL2 (Ubuntu 24.04.5 LTS) on Windows; the repository lives under the Linux home folder, not `/mnt/c`.
- CPU: Intel Core i9-14900HX, 16 cores / 32 threads visible to WSL2.
- RAM: 15 GB visible to WSL2 (the WSL2 default cap), 4 GB swap. Free disk: about 950 GB.
- GPU: NVIDIA GeForce RTX 4070 Laptop GPU, 8,188 MiB, compute capability 8.9 (Ada). Windows driver 617.14
  (`nvidia-smi` reports NVIDIA-SMI 615.78.02, KMD 617.14); CUDA 13.4 shown by `nvidia-smi`. `/usr/lib/wsl/lib/libcuda.so`
  present. No `/dev/dri` (so no Mesa `d3d12` in WSL2: offline video will need `osmesa`, Phase 4).
  So: an R580+ class driver, CUDA 13 wheels for Phase 2 (plan 4.3).
- Python 3.12.3 (the system `python3`; venv at `.venv/`, pip 26.2.1). git 2.43.0, gh 2.101.0 (logged in; `gh auth setup-git` done).
- All apt packages of plan 4.4 were already installed (git, curl, build-essential, python3-venv, python3-dev, libegl1, libgl1,
  libosmesa6, ffmpeg, gh): no `sudo` step was needed.
- Package versions after the installs of plan 4.5 (`[dev]`, `[female]`, `[physics]`, `flygym==1.2.1 --no-deps`):
  numpy 2.5.3, numba 0.67.0, llvmlite 0.49.0, pyarrow 25.0.1, pytest 9.1.1, mujoco 3.2.7, dm-control 1.0.27, dm-tree 0.1.8,
  flygym 1.2.1, scipy 1.18.1, gymnasium 1.3.0, setuptools 84.0.0 (brought in by the physics extra). `physics.available()` is true.
  Phase 2 (2026-10-01, the lean set of plan 6.2): cupy-cuda13x 14.2.0, nvidia-cuda-runtime 13.4.92, nvidia-cuda-nvrtc 13.4.92,
  cuda-toolkit 13.4.2 (the meta-package), cuda-pathfinder 1.8.2; no cuBLAS, cuRAND, cuSOLVER, cuFFT or nvJitLink wheels.
  Not installed yet: playwright (Phase 3), trimesh (Phase 3).
- The migration (2026-10-04): flygym 1.2.1 uninstalled, `flygym==2.1.0` installed with its dependencies (mujoco 3.9.0, jaxtyping,
  loguru, mediapy with ipython, tabulate; numba 0.67 and scipy 1.18 already there); dm_control, dm_tree and gymnasium are still in
  the venv but no longer used or listed. `physics.available()` is true on 2.1.

## Fresh-clone findings
(the same text is under "Known state of main" in docs/TWO_FLIES_PLAN.md)
- `origin/main` at the start: `739e791 v2.8.1: fixes for everything a fresh-clone test of v2.8.0 found (#15)`. No neuPrint
  harvest bot commit on top. History below it as the plan describes (PR #12's two workflow files, PR #11 = v2.8.0).
- Nothing in "Known state of main" differed: versions 2.8.1 in both files; `stop_loop`, `def _guard`, `OverflowError` present
  (1, 1, 1); `GF_BURST = 5`, `STILL_TICKS = 8`, `flywire.BUILD = 6`; explicit `packages` list and `pythonpath`;
  `dev = ["pytest>=7", "numba>=0.59"]`; `FLY_DATA_DIR` unset; workflows ci.yml, claude-code-review.yml, claude.yml,
  neuprint-harvest.yml; none of the **new** files existed; no `claude/two-flies-*` branch.
- One other remote branch, `claude/elegant-ramanujan-glf7g9`, is the merged source branch of PR #15 (its content is identical
  to `main`); `add-claude-github-actions-…` is PR #12's. Neither is an earlier attempt at this plan.
- Facts checked directly: 1 (16/16 twice), 2 (139,262 / 15,091,983 / 54,492,922; build 6; 15 aliases; LB3a 18), 4 (`MN9` is
  `alias of CB0701`, 2 neurons), 5 (module usage lines), 6 (token references), 7 (no new files), 8 (438 passed, 0 skipped with
  the physics extra), 10 (packages). "After the fresh-clone fixes" item 5: `Ctrl+C` on the physics game gives "Bye!" and exit
  code 0 with nothing left; item 6: Python 3.12.3, physics available.
- The venv on Python 3.12 has no setuptools by default; pip's build isolation handled the editable installs.
- pytest prints 8 warnings, all dm_control's "Setting the shape on a NumPy array has been deprecated in NumPy 2.5" (the physics
  extra, not the kit; harmless).

## Decisions
| # (section 10) | question | answer | who | date |
|---|---|---|---|---|
| 1 | operating system route | WSL2 (Ubuntu 24.04), Claude Code inside Ubuntu: the machine already matched the default | default, confirmed by the owner ("proceed with the defaults") | 2026-09-27 |
| 2 | Python version | the system python3, 3.12.3 (in the 3.10-3.12 range) | default, confirmed by the owner ("proceed with the defaults") | 2026-09-27 |
| 3 | continue before a PR is merged | wait for the merge before the next phase | default, confirmed by the owner ("proceed with the defaults") | 2026-09-27 |
| 4 | which pairs | male protagonist with a simulated FlyWire female (`--partner female`); `--partner male` also supported; `none` stays the default | default, owner: "defaults" | 2026-09-27 |
| 5 | version numbers | one minor version per merged phase: 2.9.0 for Phase 1, bumped in both files in the phase PR, SCIENCE.md headings tagged to match | default, owner | 2026-09-27 |
| 6 | song loudness rule (`SONG_MAX_HZ`) | the highest swept JO-A/B rate at which fewer than 1 % of 50 ms windows hold `GF_BURST` or more DNp01 spikes, five seeds, parts off and on | default, owner | 2026-09-27 |
| 7 | song shape | a steady drive while the male sings; a pulse-train envelope later, as a switch | default, owner | 2026-09-27 |
| 8 | song range | full within 6 mm, linear to nothing at 15 mm, real centre-to-centre mm; hand-built, provisional, no source | default, owner | 2026-09-27 |
| 9 | her DNp37 / DNp13 readouts | labelled hand-built cues with provisional thresholds, never "acceptance"/"rejection"; no effect on the male; no copulation rule | default, owner | 2026-09-27 |
| 10 | cVA channel | off (a switch) | default, owner | 2026-09-27 |
| 11 | mating status (SAG) | off | default, owner | 2026-09-27 |
| 12 | drawn flies collide | on with two or more flies; single-fly play unchanged | default, owner | 2026-09-27 |
| 13 | her walking urge | on for both flies, labelled; every claim checked against the 5.9 controls | default, owner | 2026-09-27 |
| 14 | contact arousal to pC1 for a simulated partner (`contact_pc1`) | on for a male toucher, off for a female toucher, switchable; the scripted-female path unchanged | default, owner | 2026-09-27 |
| 15 | if song barely reaches her vpoEN | report as measured; no compensating gain unless the owner asks | default, owner | 2026-09-27 |
| 16 | GPU backend, exactness and install size | CuPy with the exact ordered pull; an inexact fixed-point mode only if exactness proves impossible (opt-in, documented, asked first); the lean install (cupy-cuda13x plus the nvrtc and cudart wheels), verified (Measurements) | default; the owner: "continue" after the defaults were put to them | 2026-10-01 |
| 17 | where the GPU brains live | one GPU child process holding both brains; the game process stays on the CPU; the re-test child always uses the CPU | default, the same | 2026-10-01 |
| 18 | the fly meshes | ship a decimated GLB (about 3 MB or less) with its Apache-2.0 licence and a notice of changes; otherwise build on first use from the installed flygym | default; the owner: "merge and go with defaults" | 2026-10-03 |
| 19 | the 3-D scale in live mode | the drawn scale (about 3x real), labelled; real size in physics and replay modes | default, the same | 2026-10-03 |
| 20 | flygym version | stay on 1.2.1; a flygym 2.x spike (a Python 3.12 venv outside the repository) only on request or if 1.2.1 cannot reach 0.1x real time with two flies | default; the owner: "Defaults" | 2026-10-03 |
| 21 | topology | both flies in one MuJoCo world; "ghost" partners in separate worlds only as a labelled fallback | default, the same | 2026-10-03 |
| 22 | contact pairs between the flies | each fly's forelegs (Tibia, Tarsus1-4) and head against the partner's body, plus body to body; never `Tarsus5` | default, the same | 2026-10-03 |
| 23 | social geometry with physics bodies | real size (the physical tap replaces the 3.4 mm rule; the partner seen at real size), labelled; the single-fly physics body keeps today's drawn-scale senses | default, the same | 2026-10-03 |
| 24 | the male's body model | the same NeuroMechFly body (built from a female fly), labelled, in physics and in the 3-D view | default, the same | 2026-10-03 |
| 25 | accepting a speed lever | only if the 6.7 table stays within ±10 % for speeds and turn rates and the lure count within 1, over seeds 0-4; always switchable | default, the same | 2026-10-03 |
| 26 | MuJoCo Warp batches | no, unless the owner wants offline batch runs | default, the same | 2026-10-03 |
| – (Phase 4 order) | which of Phase 4's parts first | the reproduction gate (8.2), the pair world (8.3-8.5), record-and-replay and the video (8.7-8.9), the speed levers (8.6) last, so that the owner sees the pair played back at 1x and a video before the slow lever validations | Claude's proposal; the owner: "Defaults" | 2026-10-03 |
| – (Phase 4 jump) | a hand-built jump for the physics body | none: a giant-fibre burst stays an "escape command" with the legs standing, as v2.8.1 | Claude's proposal; the owner: "Defaults" | 2026-10-03 |
| 25 (applied) | which levers the pair adopts | `dedupe` as the pair's default: every measured number identical, the single fly's golden frames bit-identical, 1.15-1.27x faster; the rest put to the owner (SCIENCE.md 13.5) | Claude, within decision 25's rule | 2026-10-03 |
| – (the gate) | may the tool's own run on the unchanged body be the levers' baseline | yes: `../runs/p4-table.json` on this machine, SCIENCE.md 13.5 reads the levers against it | Claude's recommendation; the owner: "Go with the recommendations" | 2026-10-04 |
| 25 (applied) | `dedupe` for the single fly too | yes: the default for every physics body (`physics.DEFAULT_LEVERS`), bit-identical frames, 1.27x; `--physics-levers none` switches it off | the same | 2026-10-04 |
| – (a faster preset) | `dedupe,solver100,noslip5` for the pair | no: keep `dedupe` alone (HS in the quiet arena would rise from 19 to 27 Hz); the preset stays a switch | the same | 2026-10-04 |
| – (head-on) | two flies walking head-on slide past each other | accepted for now; a wider contact set costs speed and did not change it | the same | 2026-10-04 |
| 27 | do Phase 5 at all | yes: the owner, "do phase 5" | the owner | 2026-10-04 |
| 28 | which copy of the BANC deposit to pin | the plan's default was Dataverse V1.0; the deposit now has four published versions (1.0 of 2026-05-27 to 3.0 of 2026-07-01, titled "Publication version"); the v3 edge list and the transmitter table are byte-identical in all of them, only the metadata table differs (V1.0 MD5 `8c8babff…`, 3.0 `6275eda4…`; the GCS mirror's current one matches neither). Pinned: **version 3.0**, the publication version, as fixed and citable as V1.0; put to the owner | Claude's choice, flagged | 2026-10-04 |
| 29 | edge list | v3 (the deposit's recommended default) | default | 2026-10-04 |
| 30 | which neurons | proofread and roughly proofread, no glia, trachea or fragments; duplicates removed; every count dropped recorded | default | 2026-10-04 |
| 31 | tyramine | keep the data's label, sign +1 as a documented modelling choice, the sign switchable (+1, 0, -1) | default | 2026-10-04 |
| 32 | fru/dsx labels copied from the male | no | default | 2026-10-04 |
| 33 | the gain rule | sweep including the data-derived prior; the smallest gain at which the classic experiments pass on the mean without a runaway, stated before the sweep; calibrated on the classic six only, the extended and genetic ones held out | default | 2026-10-04 |
| 33 (applied) | the default gain | no gain met the rule; **1.0** shipped (the paper's value, no runaway, 2 of 6 classic as FlyWire's female) | the owner: "sure gain 1" | 2026-10-04 |
| 28 (confirmed) | version 3.0 of the deposit | confirmed | the owner: "version 3.0" | 2026-10-04 |
| 20 (applied) | the flygym 2.x migration as the route to real-time physics | go: a time-boxed spike after Phase 5's merge, on its own branch; the physics extra moves to flygym 2.1 and MuJoCo 3.9 (Python 3.12 or newer), the body re-validated against the gate; the physics golden hashes will change and are re-saved once the owner has seen the numbers | the owner: "route to real time" | 2026-10-04 |

## Measurements
All on 2026-09-27, this machine, `nice -n 10`, one job at a time, machine otherwise idle (load average under 1.5).

### Tests (plan 4.7)
- `python -m pytest -q` on the fresh clone: **438 passed, 0 skipped**, 24 s (the two flygym tests run: the physics extra is installed).
- With `tests/test_golden_single_fly.py` added: **449 passed**, 35 s.
- CI on PR #16 (ubuntu-latest, `[dev]` only, no flygym): Python 3.10 with NumPy 2.2.6 and numba 0.67.0: **446 passed,
  3 skipped** (the two physics tests and the golden physics configuration); 3.11 and 3.12 green too. So the synthetic golden
  hashes made here with NumPy 2.5.3 reproduce under NumPy 2.2.6 on Python 3.10: no version-dependent hash set is needed
  (plan 4.9 item 3).
- CI on PR #17 (2026-10-01, the same matrix): **547 passed, 3 skipped** on 3.10, 3.11 and 3.12 (the same three tests); here
  550 passed, 0 skipped with the physics extra.
- CI on PR #18 (2026-10-01, the same matrix, no CuPy): **570 passed, 42 skipped** on 3.10, 3.11 and 3.12 (the 39 GPU tests and the
  three physics ones); here 612 passed, 0 skipped with CuPy and the physics extra.
- CI on PR #19 (2026-10-03, the same matrix): **577 passed, 44 skipped** on 3.10 and 3.11, 576 and 45 on 3.12 (one more skip there,
  not identified from the quiet log); here 620 passed, 5 skipped (the browser tests, no Chromium yet).

### Validated experiments, the baseline JSONs (plan 4.7; game profile, five seeds, numba)
| run | file (work folder) | result |
|---|---|---|
| male, parts off | `../runs/p0-male-game.json` | 40/40 readouts, **16/16**, fragile: Smell of vinegar (MBON11) |
| male, parts on | `../runs/p0-male-game-parts.json` | 40/40 readouts, **16/16**, none fragile; after the stimulus 4 of 15 leave a runaway loop on some seed |
| female, parts off | `../runs/p0-female-game.json` | 15/26 readouts, 3/13 experiments, 3 n/a (song motor neurons, fruitless silenced, doublesex silenced) |
| female, parts on | `../runs/p0-female-game-parts.json` | 13/26 readouts, 1/13 experiments, the same 3 n/a; 3 of 12 leave a runaway loop on some seed |

The female's rows are a comparison against the male's ranges (docs/SCIENCE.md 9.3-9.4), not a test; every later phase must
reproduce all four files exactly (`tools/compare_experiments.py`).

### The female build (plan 4.7)
- Downloaded 101 MB + 32 MB into `data/flywire-src/`, built `data/flywire-v783.flyb.gz` (44.5 MB) in **30 s**; loads in 0.4 s.

### The games (plan 4.7; checked through the API, then looked at by the owner in a browser on 2026-09-27: "the games look fine")
| game | what was seen |
|---|---|
| `fly_game.py` | serves at once; with the scripted female added (the `female` action) the male walks and steers ("steering via DNa02/DNg13 + walking urge"), about 60,000 events/s, real time |
| `fly_game.py --female` | dataset `flywire:v783`, sex female, the 12-item checklist (no groom, sound, wall, court, genetics); walks and steers |
| `fly_game.py --body physics` | body `physics`, "leg physics" in `whats_real().hand_built`; the loop's own average read 0.2× real time in its first seconds (see the benchmark for the number to quote) |
| `Ctrl+C` on `--body physics` (the 1.7 recipe) | "Bye!" then **exit code 0**, nothing left in the session: the baseline for 5.12 and 8.11 |

### Benchmark (plan 4.8; `../runs/p0-bench.json`, the same numbers in docs/SCIENCE.md 9.6)
`tools/bench_two_flies.py`: game profile, `dt` 0.5, numba, busy input, 3 s after 300 ms warm-up, one process per row,
`nice 10`, idle machine. Real-time factor (RTF) = simulated / wall time.

| row | ms per step | RTF | events/s | peak memory |
|---|---|---|---|---|
| brain, male, parts off | 0.138 | 3.62 | 160,781 | 384 MB |
| brain, male, parts on | 0.246 | 2.03 | 215,625 | 445 MB |
| brain, female, parts off | 0.130 | 3.85 | 141,675 | 580 MB |
| brain, female, parts on | 0.295 | 1.70 | 265,713 | 705 MB |
| pair (both at once), parts off | male 0.192 / female 0.178 | 2.61 / 2.82 | as above | 384 / 580 MB |
| pair (both at once), parts on | male 0.307 / female 0.328 | 1.63 / 1.53 | as above | 445 / 704 MB |

| row | RTF | tick p50 / p99 | setup | peak memory |
|---|---|---|---|---|
| game, male, drawn, 400 ticks | 3.14 | 7.9 / 10.5 ms | 1.6 s | 813 MB |
| game, female, drawn, 400 ticks | 4.28 | 5.7 / 6.9 ms | 2.9 s | 1,624 MB |
| game, male, physics, 80 ticks | 0.216 | 112 / 147 ms | 3.5 s | 1,509 MB |

- **The pair rows are the Phase 1 number:** two busy brains side by side run at 2.6-3.4x real time (parts off, both runs) and
  1.4-1.6x (parts on) on this CPU, well above the 0.7x target of plan 3.3, before any pipe or lockstep cost.
- The plan's script measured memory with `ru_maxrss`, which never falls within a process and which Linux hands to spawned
  children, so the first run reported the parent's high-water mark on every pair row (853 MB). The committed script runs every
  row in its own child and reads `VmHWM`. The first run's timings (`../runs/p0-bench-first-run.json`) differ from the second's
  by up to 23 % on the pair rows (male 0.156 → 0.192, female 0.146 → 0.178 ms per step, parts off) and the male parts-off
  brain row read 0.203 there against 0.138 here: treat the first decimal of ms per step as the precision, and the pair's
  cost over a lone brain (11-39 % in the second run, −23 % to +33 % in the first) as within the run-to-run spread.
- Physics: 0.22x real time here (the plan's reference machine: 0.086-0.113), 112 ms per tick.

### Phase 1: cell counts behind the social channels and the female's readouts (plan 5.5, 5.6; checked 2026-09-27)
`Connectome.count(spec)` on the kit's files (male / female): `prefix:JO-A,prefix:JO-B` 138 / 359; `LC10a` 275 / 234; `LC11` 143 / 127;
`LC4` 126 / 104; `LPLC2` 185 / 210; `LgLG1a,LgLG1b` 270 / **0**; `ORN_DA1` 204 / 126; `prefix:pC1_` 148 / 10; `pIP10` 2 / **0**;
`DNp13` 2 / 2; `DNp37` **0** / 2; `vpoEN` 4 / 4; `AN_SMP_2` (SAG) 0 / 2; `prefix:BM_InOm` 745 / 1,113; `DNp01` 2 / 2; `MDN` 4 / 4.
Female only: the pC2l spec `AVLP567,AVLP568,AVLP569,AVLP570,CL313,SIP200f,SIP201f,!body:720575940610359758` 38 (39 without the
exclusion), `DNp55` 2, `oviDNa_a,oviDNa_b,oviDNb` 6, `DA1_lPN` 15, `aSP-g1,aSP-g2,aSP-g3A,aSP-g3B` 23, `M_lvPNm43,M_lvPNm45` 10.
Wiring spot checks (`--female --inputs`): pC1a → DNp37 382 synapses (19.0 % of its 2,008 input synapses; the five pC1 types together 530, 26.4 %), vpoEN → DNp37 169, CL313 → DNp13 678 and
AVLP569 142 (the pC2l types), vpoEN → DNp13 275, AN_SMP_2 → pC1 872. All as the plan's channel table says.

### Phase 1 steps 1-2 (2026-09-27)
- Step 1 (`89d6f78`, FlyAgent): 449 passed; real-data golden compare 18 unchanged.
- Step 2 (`ae3d656`, the BrainIO seam): 461 passed (12 new in `tests/test_brainio.py`); real-data golden compare 18 unchanged;
  the three golden configurations autopilot, zap_mdn and silence_and_watch give the same hashes through a process brain
  (`Game(brain_procs="on")`). `Ctrl+C` on `fly_game.py --brain-procs on`: "Bye!", exit code 0, nothing left (the brain child and
  multiprocessing's resource tracker both gone).
- **Pipe cost per tick** (real male, drawn body with the scripted female, 400 ticks, two runs each): inline p50 7.01-7.06 ms,
  p99 8.75-9.35, mean 7.10-7.19; process brain p50 7.56-7.81, p99 9.98-10.11, mean 7.66-7.87. The pipe costs about **0.6 ms per
  tick** (p50 +0.65, p99 +0.99, mean +0.62): under the plan's 1 ms line, so pipes stay and no shared memory is used.

### Phase 1: the song level (plan 5.9 item 1, decision 6; 2026-09-27)
`tools/song_startle.py` (new): the sound cells `prefix:JO-A,prefix:JO-B` driven at 10-100 Hz, 5 s per rate and seed after
100 ms, seeds 0-4, game profile, the giant fibre `DNp01` binned into 25 ms ticks, windows of two ticks; a burst is
`game.GF_BURST` = 5 or more spikes. Share of windows with a burst (largest window in brackets), male, parts off / on:

| JO-A/B Hz | 40 | 50 | 60 | 70 | 80 | 90 | 100 |
|---|---|---|---|---|---|---|---|
| parts off | 0 % (4) | 0.20 % (5) | 0.10 % (5) | **0.50 % (5)** | 1.21 % (6) | 3.12 % (6) | 6.73 % (7) |
| parts on | 0 % (4) | 0.20 % (5) | 0.10 % (5) | **0.80 % (5)** | 2.21 % (6) | 4.42 % (6) | 10.15 % (6) |
| GF Hz per cell (off / on) | 14.0 / 14.0 | 16.8 / 16.8 | 19.4 / 19.4 | 22.6 / 23.2 | 25.7 / 26.4 | 27.9 / 29.0 | 31.0 / 32.2 |

**`SONG_MAX_HZ` = 70 Hz** by the rule (the highest swept rate under 1 % in both settings; 80 Hz crosses it). At 70 Hz every burst
window holds exactly 5 spikes (the threshold), 5 of 995 windows with the parts list off and 8 with it on. Under the same drive
the female's giant fibre never bursts: largest window 4 (parts off) or 3 (parts on) at 100 Hz, 0 % at every rate; her GF mean
reaches 21.2 / 13.1 Hz per cell at 100 Hz. (`../runs/p1-startle-male.json`, `../runs/p1-startle-female.json`; 114 s.)

### Phase 1: what the sweep does to each fly's readouts (plan 5.9 item 1; `fly_brain.py --sweep`, 5 seeds, `../runs/p1-song-*.json`)
Male (parts off and on alike): DNp01 4.2 → 28.0 Hz per cell from 10 to 100 Hz; pC1 0.0 and vpoEN 0.0 at every rate.
Female, parts off: DNp01 0.0 → 26.5 Hz per cell; **vpoEN, pC2l (38 cells), DNp37 and DNp13 stay at 0.0 Hz at every rate up to
100 Hz**; DNp55 0 → 4.8 Hz (from 60 Hz on). Parts on: DNp01 0 → 15.8; the four decision readouts 0.0 throughout; DNp55 0 → 4.8.
So in this data his song reaches her Johnston's organ and her giant fibre (never to a burst) but not her song-tuned neurons
(vpoEN), her pulse-song detectors (pC2l) or the two motor commands (DNp37, DNp13): the Johnston's-organ synapses Baker et al.
(2022) found under-detected in FlyWire v274 look under-detected in v783 too. Reported as measured; no compensating gain
(decision 15).

### Phase 1: the pair experiments (plan 5.9 item 2; commit `468f757`; `../runs/p1-pair-{female,male}[-parts].json`)
Five seeds, game profile, parts off and on. Ranges set from the measurement (provisional). Female: F1 (song at 70 Hz):
vpoEN, pC2l, DNp37, DNp13 all 0 on every seed, DNp01 7-20; F2 (pC1 80 Hz): DNp37 72-86; F3 (pC2l 80 Hz): DNp13 105-113;
F4 (ORN_DA1 80 Hz): DA1_lPN 99-102, aSP-g/pC1d/pC1e 0; F5 (SAG 60 Hz): pC1 11-33, DNp37 17-45. Male M1 (song 70 Hz): DNp01 19-25.
Summary lines: female 13/14 readouts, 5/6 experiments (M1 on her reads 18.4 / 9.0 Hz, below his range); male 5/7, 2/4 (F2, F5 n/a on him;
F1 fragile, F3 DNp13 0 Hz, F4 DA1_lPN 207 Hz as comparisons). `--pair-experiments` refuses the 12 modes that run no experiments (exit 2)
and matches `--only` against the pair list. Written into docs/SCIENCE.md 10.4.

### Phase 1: the two-fly game's speed and memory (plan 5.9 item 4; `../runs/p1-bench-pair-game.json`; commit `6008897`)
`tools/bench_two_flies.py --only pair-game` (male + FlyWire female, brains in processes, channels seen/song/contact/collide,
400 ticks after 20): parts off dt 0.5: RTF 2.87, tick p50 8.66 / p99 10.28 ms, parent 1,655 MB, children 409 + 630 MB; parts off
dt 1.0: RTF 4.27, 5.77 / 7.58 ms; parts on dt 0.5: RTF 1.88, 13.34 / 16.0 ms, parent 1,721, children 470 + 714 MB; parts on dt 1.0:
RTF 2.60, 9.49 / 12.28 ms. Target (plan 3.3): 0.7x. In docs/SCIENCE.md 10.2.
Rerun 2026-10-01 on the idle machine (load average 0.02 at the start; `../runs/p1-bench-pair-game-idle.json`): parts off dt 0.5:
RTF 2.70, 8.99 / 14.17 ms; parts off dt 1.0: 4.16, 5.93 / 7.98; parts on dt 0.5: 1.74, 14.31 / 18.11, parent 1,742 MB; parts on
dt 1.0: 2.33, 10.11 / 17.7; memory otherwise within 1 MB of the first run. The first run had started two minutes after the
pair-experiment runs, at a 1-minute load average of 5.3 (not stated in 10.2 until now); the idle rerun is the slightly slower
one (medians 3-7 % longer, 99th percentiles 5-44 %), so the difference is the run-to-run spread of 9.6, not the load. Both
runs are in docs/SCIENCE.md 10.2.

### Phase 1: the pair over 90 s with its controls (plan 5.9 item 3; `tools/pair_courtship.py`; `../runs/p1-pair-courtship.json`; 13 min)
Five seeds, parts off, brains in processes; full / her urge off / song off / all off. Near 15 mm: 70-84 / 59-88 / 70-82 / 4-17 s. He sings
60-74 / 63-80 / 58-74 / 0 s; taps 10-15 / 9-16 / 11-16 / 0 per min. She hears 39-48 Hz (78-93 % of ticks) / 43-53 / 0 / 0. Her vpoEN and
DNp37 0 everywhere; DNp13 0-1.0 / 0-0.4 / 0-1.2 / 1.1-3.4 Hz. Her speed while he sings vs not: 4.0-7.3 vs 5.8-8.7 / 0.4-0.9 vs 0.8-1.9 /
5.1-6.7 vs 6.3-8.1 / — vs 6.0-7.8 mm/s: the slowing survives the song channel being off, so it is his bumping and tapping, not song
(no song effect measured; decision 15 stands). Bursts his / hers per seed: 3,1,2,0,0 / 2,0,1,2,1; urge off 0 / 13,6,3,0,0; song off
3,2,3,0,1 / 0,1,0,2,1; all off 0 / 0. Written into docs/SCIENCE.md 10.5. (The run started before commit 9fa090d, so its JSON lists "cues"
among the channels and its escape-event counts lack the per-fly tag; the burst counts per fly are from each fly's own giant fibre.)
Rerun on 2026-10-01 with the tool as committed (`../runs/p1-pair-courtship-rerun.json`; load average 0.7 at the start, 34-39 s of wall
time per run, 12.5 min in all): every per-seed value of every measure (near, facing, singing, taps, hearing, speeds, her readouts,
bursts) is identical to the 2026-09-27 run; new in it, each fly's giant-fibre rate and the escape jumps counted per fly as entries
into the escape mode (his / hers per seed: full 1,1,1,0,0 / 1,0,1,1,1; urge off 0 / 7,3,2,0,0; song off 2,1,2,0,1 / 0,1,0,2,1; all
off 0 / 0; a burst within the second after a jump is a burst, not a jump). In docs/SCIENCE.md 10.5.

### Phase 1: the validated experiments against the Phase 0 baseline (plan 1.9)
`fly_brain.py --profile game` (male, parts off and on) and `--female --profile game` (parts off and on) rerun at the end of the phase
(`../runs/p1-*-game*.json`): `tools/compare_experiments.py` reports **identical** to `../runs/p0-*.json` for all four (every readout's
mean and per-seed rates, ok and fragile flags, after-stimulus activity).

### Phase 1: the third review and its fixes (2026-10-01)
The review of steps 6-8, the tools and SCIENCE.md section 10 launched on 2026-09-27 never ran (the session's usage limit stopped its
five agents at once), so it was rerun on 2026-10-01 with seven lenses (the per-fly API and state, the page, the pair scenario and
checks, the tools, the documents, CI compatibility on Python 3.10 with NumPy 2.2, and the acceptance criteria of 5.10-5.12 against
the rules of 1.2-1.6), two refuters per finding: 40 candidate findings, 20 confirmed (17 distinct), 20 refuted, most on materiality.
Confirmed and fixed (`8c5ec16`, `41238fa`, `c9813b6`, a regression test per code point): the walking urge missing from each fly's
state entry (the page toggled fly k's by fly 0's value); the spike recording started on fly 0's brain only (`/api/spikes?fly=1` an
empty file); the paused state's entries not mirroring the top level; the pair courtship scenario taking fly 0 as the male (wrong with
`--female --partner male`, meaningless with two males); the "seen" pair items ticking on a sugar drop or a post; "sang" ticking at any
distance; two males told to add a female; the pathway explorer, the type search and the ontology search asking fly 0 whatever the
focus; the male page's NeuronBridge tooltips blanked; an unbounded spike buffer in the 2-D fallback; five works cited in SCIENCE.md
10 without a reference entry; "382 synapses from pC1a, 26 %" (19.0 %; the five pC1 types 530, 26.4 %); 400 ms for a 500 ms window;
the two-fly speed stated without the machine load (measured again, idle: above); `--partner male` ticked in the PR body without a
recorded run (below); the PR body's unfilled count and a session link. Refuted on materiality but fixed as well, since each was
cheap: `?fly=²` answering 500; the event log not saying whose event; the F-key message wiped by the next tick; the keyboard dead
after the fly menu; a pre-start tap counted by the pair measure; pair items listed for channels that are off; escape events counted
from a 200-entry log; "over every seed" for a pooled rule; the receivers' total input missing from 10.3; README's 2.8x against 2.9x;
a 5.2 mm collide figure that matched nothing; the missing `tests/test_agent.py`; API.md's "world actions ignore fly"; no on-screen
label on her cues. Documented, not changed: a grow or parts rebuild of a brain in its own process runs in that child while the game
waits (API.md says so now; the inline brain's background rebuild is unchanged). Left as it is: per-rate sweep numbers depending on
the rates before them (the kit's `--sweep` does the same; not a Phase 1 change). CI compatibility: no finding. 550 tests after the
fixes; 18 real golden hashes unchanged.

### Phase 1: the end-of-phase game checks (plan 5.12; 2026-10-01, on the final code)
The 1.7 `Ctrl+C` recipe (SIGINT to the game's own Python, then `ps --sid`), each game read through the API twice, 6 s apart, first:
`fly_game.py --partner female`: both flies in the state, he in court mode, "Bye!", exit code 0, nothing left (the game, its two brain
children and multiprocessing's resource tracker all gone); `--partner male`: both male brains in their own processes (the two flies'
readouts differ at the same t; the second brain's seed is the first's + 1000, SCIENCE.md 10.2), "Bye!", exit code 0, nothing left;
`--body physics`: the fly walked (t 1.65 s after about 7 s of wall time), "Bye!", exit code 0, nothing left. A first attempt from a
non-interactive script left all three games ignoring SIGINT (a background job of a script has SIGINT ignored, as plan 1.7 warns):
with job control on (`set -m`) the recipe behaves as in the terminal. (`../runs/p1-ctrlc-*.log`.)

### Phase 2: the CuPy install and checks (plan 6.2; 2026-10-01)
`pip install -c ../runs/constraints.txt cupy-cuda13x "cuda-toolkit[nvrtc,cudart]==13.*"` with numpy 2.5.3, numba 0.67.0 and llvmlite
0.49.0 pinned: cupy-cuda13x 14.2.0, nvidia-cuda-runtime 13.4.92, nvidia-cuda-nvrtc 13.4.92, cuda-toolkit 13.4.2, cuda-pathfinder 1.8.2
(about 70 MB of wheels); NumPy, numba and llvmlite unchanged. `cupy.show_config()`: CUDA driver 13.4 (13040), runtime 13.2 linked to
CuPy / 13.4 installed, NVRTC 13.4, device NVIDIA GeForce RTX 4070 Laptop GPU, compute capability 8.9, 7,050 of 8,187 MB free; the cuRAND
and cuSOLVER wheels absent as intended (the lean set), cuBLAS and cuSPARSE reported available. The plan's check: a `RawKernel` compiled
with `--fmad=false` ran (0.86 s including NVRTC), and stream capture into a CUDA graph and its launch worked. The NVRTC cache lives in
`~/.cupy/kernel_cache` unless `CUPY_CACHE_DIR` is set. After the install: **550 passed**, and `tools/golden_hashes.py --compare`
**18 unchanged** (nothing moved).

### Phase 2: the GPU brain's equality on the real data (plan 6.6; 2026-10-01; `../runs/p2_real_equality.py`, `../runs/p2-eq-*.json`)
Game profile, seed 0, the benchmark's busy input, 2 s = 4,000 steps stepped a tick (50 steps) at a time on numba and on the GPU
side by side, every step's spike list compared, then every state array byte for byte (`v`, `g`, `thr`, `spike_count`, `std_x`,
`std_t`, the release accumulators, the tone levels and gains, the ring, `w`) and `t`, the quiet flag, the spike total and the
pending flags: **identical** in all four runs: male parts off 358,953 spikes; male parts on 440,318; female off 242,340; female on
431,990. The depression table has one entry in the game profile (no depression: `std_u` 0); `math.exp` equalled `np.exp` on every
entry. Build: numba 0.3-0.8 s, cupy 2.0-3.1 s (the first NVRTC compile is cached). The engine's own tests: 32 passed in 3.3 s.

### Phase 2: the 16 validated experiments on the GPU (plan 6.6, 6.9; 2026-10-01; `../runs/p2-experiments.sh`, `../runs/p2-*-cupy.json`)
`fly_brain.py --profile game --backend cupy` (male, parts off and on) and `--female --profile game --backend cupy` (parts off and on):
`tools/compare_experiments.py` against `../runs/p0-*.json` reports **identical** for all four (every readout's mean and per-seed rates,
the ok and fragile flags, the after-stimulus activity); the closing lines name the integrator "the GPU (CuPy, --backend cupy)".
About 15-20 s of wall time per run on the first engine.

### Phase 2: the two-fly game on the GPU through the brain server (plan 6.5; 2026-10-01; `../runs/p2-ctrlc-gpu.log`)
`fly_game.py --partner female --backend cupy` with the 1.7 `Ctrl+C` recipe: "Brain integrator: the GPU (CuPy) in the brain process";
"Partner: flywire:v783 (female), its brain in the one brain process, as is the first fly's"; one brain child (plus multiprocessing's
resource tracker) in the session, both flies in the state, he in court mode after 7 s; "Bye!", exit code 0, nothing left.

### Phase 2: where a GPU tick's time went before the speed work (2026-10-01; the profile script in the session scratchpad)
The same input, 80 ticks, 40 monitors with 25 ms bins as the game has, plasticity on, parts off, ms per 25 ms tick: male 5.92 in all
(4.2x real time), of which the device stream (the input upload, the ten-step graph, the two downloads) 4.26, `plasticity.step`
0.53, the per-launch weight comparison 0.31, the monitors 0.17, the random draws 0.19, the rest of the replay loop 0.46; female 7.68
(3.3x): stream 5.77, plasticity 0.53, weights 0.49, monitors 0.16, draws 0.20, loop 0.53. The equality script's own numbers (no
monitors): male 0.139 ms per step on the GPU against numba's 0.148; male with parts 0.239 against 0.257; female 0.180 against 0.122;
female with parts 0.350 against 0.272. So the first engine runs the male at numba's pace and the female slower, well short of the
6.9 target (both brains in 8 ms per tick): the device time is launches (nine kernels per step) and passes over the neuron arrays
rather than the propagation, the host time is per-step Python. The speed work (merged kernels, one download per chunk, a warp per
hit target, a per-block plasticity step, the monitors and weights checked only when they can change) follows, with the equality
script and the tests as the guard.

### Phase 2: the speed work on the GPU brain (2026-10-01; the profile script's numbers, ms per 25 ms tick, male / female)
Game profile, busy input, 40 monitors with 25 ms bins, plasticity on, parts off unless said. A race fixed first: the first engine
made its device arrays on CuPy's default stream and filled them on its own non-blocking one, which are unordered, so next to
another process's GPU load an initialising memset could land after an upload (2 of 50 runs beside a GPU hog gave different
spikes: a brain with every threshold wiped to zero fired everywhere at step 0); every device operation now runs on the engine's one
blocking stream (0 of 110 runs). Then, in order: the start 5.92 / 7.68 (device stream 4.26 / 5.77); the stream fix 5.77 / 7.95;
the kernels merged (a scatter kernel for the host's noise counts and forced marks, one dense kernel that also appends the spikes
to the log), the send a block per spiking neuron, the hit list gathered with warp-aggregated atomics, the pull a warp per hit
target with the kicks still added one at a time in edge order: 4.62 / 4.93 (device 1.94 / 2.06); pinned host buffers and one
download per chunk: 4.44 / 4.24; the host's bookkeeping per chunk (`MushroomBodyPlasticity.step_block`, exact and tested
field by field; `spike_count`, the APL tally, the tone deposits and the refractory lists per chunk; monitors only when a bin
can close; the weight subsets through a scatter kernel, the plastic ones only after a block that changed them): 3.74 (male);
chunks of 20 steps where a block starts (three launches per tick, two graphs) and the gain uploaded at block starts: **4.22 /
3.95**, with the parts list on 5.11 / 6.43. The device's share is now 1.9 ms per brain per tick (about 38 µs per step: the dense
pass 12, the pull 10, the send 8, the hits 3, the rest 5). Per step without monitors (the equality script): male 0.083 ms on
the GPU against numba's 0.143 (6.0x against 3.5x real time), female 0.092 against 0.124, male parts on 0.116 against 0.252,
female parts on 0.149 against 0.260. Both brains back to back in one process: about 8.2 ms per tick with the parts list off
(3.0x real time, the 6.9 target just met) and 11.5 ms with it on (2.2x). Left for later: launching both brains' chunks before
waiting on either (the device time would hide behind the other brain's host replay; it needs a generator-shaped
`advance_steps` and a change in the brain server); narrower per-neuron arrays in the dense pass.

### Phase 2: the final measurements on the committed engine (plan 6.6, 6.7; 2026-10-01; `../runs/p2-final-measure.sh`, 1-minute load 1.1-1.3)
- Equality on the real data, rerun (`../runs/p2-eq-*.json`): **identical** in all four runs again (the same spike counts: 358,953 /
  440,318 / 242,340 / 431,990); per step, the GPU against numba, no monitors: male 0.083 against 0.144 ms (6.0x against 3.5x real
  time), male parts on 0.119 against 0.252 (4.2x against 2.0x), female 0.089 against 0.125 (5.6x against 4.0x), female parts on
  0.148 against 0.265 (3.4x against 1.9x).
- The 16 experiments with `--backend cupy`, rerun: **identical** to the Phase 0 baseline in all four runs; about 10 s per run now.
- `tools/bench_gpu.py` (`../runs/p2-bench-gpu.json`): the environment row (an RTX 4070 Laptop GPU, 36 multiprocessors, 8,188 MB,
  a 128-bit bus at 8.0 GHz effective, a measured 225 GB/s device-to-device copy; CuPy 14.2.0, CUDA runtime 13.2 / driver 13.4);
  the steps rows (game profile, the busy input, 2 s after 300 ms of warm-up, stepped a tick at a time, microseconds per step,
  median / 99th percentile): male numba 131 / 238 against cupy **61 / 78** (3.8x against 8.2x real time); male parts on 229 / 269
  against 90 / 105 (2.2x against 5.6x); female 108 / 127 against 69 / 81 (4.7x against 7.3x); female parts on 248 / 296 against
  137 / 280 (2.0x against 3.7x). One `step()` at a time on the GPU (its own launch, no graph) costs 500-630 µs: the API's single
  step is for tests, the chunks are the way to run.
- The two-fly game through the brain server (`tools/bench_two_flies.py --only pair-game --backend cupy`, 400 ticks after 20, in
  `../runs/p2-bench-gpu.json`; the CPU's rows rerun the same minute into `../runs/p2-bench-pair-game-cpu.json`): parts off dt 0.5:
  GPU RTF 2.03, tick 11.5 / 20.6 ms (median / 99th) against the CPU's two processes 2.78, 8.9 / 11.2; dt 1.0: 2.09, 11.3 / 19.6
  against 4.03, 6.1 / 9.0; parts on dt 0.5: 1.46, 16.2 / 22.8 against 1.80, 14.0 / 16.5; parts on dt 1.0: 1.37, 18.1 / 22.8
  against 2.63, 9.5 / 11.1. The GPU's game process used 11-15 % of a core, the GPU 38-40 % (nvidia-smi sampled), 1,650-1,670 MB
  of device memory for both brains. **So the first server runs the two brains one after the other and loses to the CPU's two
  parallel processes** although each brain alone is faster on the GPU: the 6.9 target of 8 ms of brain time per tick for both is
  missed by a little (about 8.7 ms sequential) and the 99th-percentile tick stays under 25 ms. The fix is to launch both brains'
  chunks before waiting on either (one brain's device time behind the other's host work, the two graphs on two streams): the next step.

### Phase 2: the two brains overlapped in the brain server (plan 6.5; 2026-10-01; `../runs/p2-bench-pair-game-gpu2.json`)
A chunk is now a launch and a collect; a GPU brain's tick is a generator that pauses after each chunk's launch; the server resumes
its brains' ticks in turn (A launches, B launches, A collects and replays and launches again while B's chunk runs on its own
stream). `tools/bench_two_flies.py --only pair-game --backend cupy` (400 ticks, 1-minute load 2.7 at the start), real-time factor
and tick median / 99th percentile, before (one brain after the other) and after: parts off dt 0.5: 2.03, 11.5 / 20.6 ms ->
**3.56, 6.95 / 8.41 ms** (the CPU's two processes the same hour: 2.78, 8.9 / 11.2); parts off dt 1.0: 2.09, 11.3 / 19.6 -> 4.26,
5.70 / 8.98 (CPU 4.03, 6.1 / 9.0); parts on dt 0.5: 1.46, 16.2 / 22.8 -> 2.55, 9.81 / 12.0 (CPU 1.80, 14.0 / 16.5); parts on
dt 1.0: 1.37, 18.1 / 22.8 -> 2.65, 9.25 / 13.6 (CPU 2.63, 9.5 / 11.1). The game process 19-28 % of a core, the GPU 23-35 %. So
the 6.9 targets hold with the parts list off (3.6x; the 99th-percentile tick 8.4 ms) and the GPU game beats the CPU's in every
row; with the parts list on 2.6x, because each brain's host replay (the tone deposits and the local release) is longer than the
other brain's device time. Equality on the real data rerun on this code: identical in all four runs. Tests: 127 in the files
touched; the whole suite below.

### Phase 3: three.js vendored, the meshes' licence, the installs (plan 7.1, 7.2, 7.6; 2026-10-03)
- three.js: the registry's latest is still 0.186.1 (r186, MIT). The tarball (4,648,523 bytes) was checked before anything was
  copied: its SHA-512 equalled the registry's `dist.integrity` (`sha512-blFeqb49...`) and its SHA-1 the `dist.shasum`
  (`6d50f70c...`); SHA-256 `8cd068708ea44f2c73c944b1cead2ba2f0d5c15c8fc194e5700f4e4f4a033fe7`. Copied unchanged into
  `virtual_fly/web/vendor/three/`: `three.module.js` (662,772 bytes), `three.core.js` (1,458,113), `addons/loaders/GLTFLoader.js`
  (117,570), `addons/controls/OrbitControls.js` (40,755), `addons/utils/BufferGeometryUtils.js` and `SkeletonUtils.js`, `LICENSE`
  (the sizes the plan gave); every file's SHA-256 is in `virtual_fly/web/vendor/three/VERSION.txt`. The import map is in
  `index.html`; nothing loads it until the 3-D view is switched on.
- The meshes' licence (7.2, verify first): the installed flygym 1.2.1 wheel's `LICENSE` is the Apache-2.0 text (the unfilled
  template), its METADATA says `License: Apache-2.0`, and the repository's `LICENSE` on GitHub (NeLy-EPFL/flygym, main) is the same
  Apache-2.0 text; the repository root has no NOTICE and no separate data licence. So the meshes may be converted and shipped
  with the licence text, a notice of the changes and the attribution (decision 18).
- Installs (PyPI, with `../runs/constraints.txt` pinning numpy 2.5.3, numba 0.67.0, llvmlite 0.49.0, all unchanged after): trimesh
  5.1.0 (the plan's tested version; 5.1.1 is the newest) and fast-simplification 0.2.0 for the decimation; playwright 1.63.0 (the
  Python package only; the Chromium download is on the plan's ask-first list and waits for the owner's word). setuptools in the
  venv is 84.0.0 (`bdist_wheel` built in since 70.1): no upgrade needed for the wheel check.

### Phase 3: the fly model files (plan 7.2, 7.3; 2026-10-03; `tools/build_fly_model.py`, 1.0 s)
The compiled NeuroMechFly model is in millimetres already (the MJCF scales the metre STLs by 1,000; the thorax stands 1.497 mm
above the floor), so nothing was rescaled. Triangles 502,781 -> **58,293** per fly (the head 8,000, each eye 2,500, the thorax
4,000, abdomen segments 1,500, wings 1,499, leg segments 300-900; nothing refused decimation; no winding flipped).
`nmf_fly.glb` 1,467,684 bytes; `nmf_gait.bin` 125,580 bytes, shape (65, 69, 7); `nmf_gait.json` 5,545 bytes; the licence and the
notice. The stride is 83.33 ms (the CPG's 12 Hz); frame f sets the 42 leg joints to the Walker's `neutral + step(2 pi f / 64 +
the leg's tripod bias)`, the bias convention checked against the kit's CPG run at full drive (settled phases equal the bias row to
3e-12 rad), then `mj_kinematics`; frame 64 is the Walker's neutral stance (a hand-built choice, labelled). Hinge points in the
thorax frame (mm): wings (-0.544, +-0.370, 0.181), the abdomen (-0.862, 0, 0), the proboscis (0.459, 0, -0.274). Tests: the files'
consistency (3), the packaging (9: every web folder listed; a wheel from a copy carries the 8 vendored and 5 model files with no
warning; setuptools 84 builds it).

### Phase 3: the page (plan 7.3-7.6; 2026-10-03)
`virtual_fly/web/arena3d.js` (the 3-D dish with arena.js's interface and the camera presets), the toggle, the badge, the `whats_real`
lines, the static pins and `tests/test_browser.py` written; the full suite 620 passed and 5 skipped (the browser tests: no Chromium on
this machine yet), the 18 real golden hashes unchanged. The page's new module parses (esprima's ES2017 grammar; the two page files it
rejects use optional chaining it predates and were not touched); its brace balance checked; no JavaScript engine or browser ran it.
**Not done, by the plan's rules:** the Chromium download for Playwright (`python -m playwright install chromium`, about 150 MB, plus
`install-deps` with sudo on WSL2) is on the ask-first list of 1.6 and needs the owner's word; the frame-rate target (7.5, 7.7: 50 fps at
1080p with two flies) is a number from a headed browser on the owner's screen. So 7.7's first, third, sixth and seventh criteria wait
for them; the others hold.

### Phase 3: the page in a real browser (plan 7.5-7.7; 2026-10-03)
On the owner's word the Chromium of Playwright 1.63.0 was downloaded (`python -m playwright install chromium`: Chrome Headless Shell
153.0.8010.12 plus the full Chromium and ffmpeg, 658 MB under `~/.cache/ms-playwright`). Its binary wanted three system libraries this
WSL2 Ubuntu lacks (`libnspr4.so`, `libnss3.so`, `libnssutil3.so`; `playwright install-deps` would install 32 packages with sudo): the
two Debian packages were fetched with `apt-get download` (no root) and unpacked into the session's scratch folder, and the tests run with
`LD_LIBRARY_PATH` pointing at them; nothing on the system was changed. The owner can make Chromium usable without that path with
`sudo .venv/bin/python -m playwright install-deps chromium`.
`VF_BROWSER_TESTS=1 pytest tests/test_browser.py`: **5 passed** in 17 s, the page's first run in any browser: the default page fetches
nothing under `vendor/three/` or `models/`; the 3-D button gives a WebGL2 context, the canvas, the badge and the fly with no console
error; with WebGL2 blocked the page stays in 2-D with the message; a two-fly game shows both flies (125,270 triangles, 151 draw calls);
the frame-rate probe reports 12 fps for one fly and 9-10 for two in headless Chromium on SwiftShader (a software renderer: not the
plan's number). Screenshots from the same headless browser (the top and the follow camera) checked by eye: both flies on the dish, the
badge and the camera menu in place. The model is 3.72 mm long at real size and is scaled by 1.93 to the drawn 7.2 mm.
The owner looked at the 3-D view in their browser on 2026-10-03 ("looks decent") and asked for colour ("it is all brown"): the flies
are now coloured by part (`PALETTE` in arena3d.js: red compound eyes, a tan head and thorax, an abdomen banded tan and dark with a dark
tip on the male, clear wings at 38 % opacity, darker legs and tarsi, dark aristae; the female a shade lighter and greyer), said to be
hand-chosen in "What's real here?" and SCIENCE.md 12. The owner then said "color looks good and it is smooth" and asked how to see the frame rate: the badge now carries a
frame-rate readout the dish updates once a second (`#fps3d`; `window.__vf3d.fps` holds the same number for the console). The owner read
it: **"the fps is at 80 or so"** (2026-10-03, the two-fly game in their own browser on this laptop; the GPU the browser used, the window
size and whether the brain map was open were not recorded): the 7.7 target of 50 met.

### Phase 4: the reproduction gate (plan 8.2; 2026-10-03; **new** `tools/physics_table.py`; `../runs/p4-table.json` and `.log`; the first run with a still lure in `../runs/p4-table-still-lure.*`)
- The protocol, reconstructed: PR #11's text and the v2.8.0 docs commit (`6a8603a`) hold no script, only the table's own row
  labels, so the tool's docstring states what it does: the body alone for the speed and turn rows (the net thorax displacement
  and the heading change over 3 s); the lure placed before the first tick at 20 mm and 70 deg (15 mm with the urge off); MDN zapped
  at 60 Hz for the full 3 s; 10 s quiet; every run a fresh game in its own process (brain seed = game seed, parts on, dt 0.5).
  **The lure wiggles** 1.5 mm across its bearing at 3 Hz: with a still lure the drawn fly with the walking urge off did not turn
  at all (0.0 deg on every seed; published 62-73), because a standing fly sees a still lure stand still (LC10a answers motion on
  the retina) and the page's own check says "wiggle the decoy"; 1.5 mm at 3 Hz gives 60-81 deg in place, facing at 0.60-0.75 s
  (published 62-73 deg, 0.48-0.8 s) and, with the urge on, facing times of 0.93-1.25 s (published 0.95-1.33); 3 mm at 3 Hz faces too
  early (0.70-0.85 s) though it reproduces seed 2's jump-and-circle (255 vs 253 deg); 1.5 mm at 6 Hz sets off escapes on three seeds.
- The run: 85 fresh games, 6 at a time, 6.7 min (1-minute load 1-12; each physics run 1.57 GB). Against the published table:
  - **drawn**: speed 4.1 / 8.1 / 13.6 mm/s (chosen 4.2 / 8.4 / 13.9: the net displacement carries the velocity ramp); turn 259 / 224 /
    177 deg/s (264 / 228 / 180); lure left 103, 105, 79, 90, -60 deg (97, 79, 253, 94, 257: seeds 2 and 4 circled after a jump under
    the v2.8.0 rule; under the burst rule seed 4 still escapes on both sides, 1 event of 7 ticks, and circles the other way; seed 2
    does not); right -95, -101, -118, -104, +68 (-94, -102, -89, -100, +82); faces the lure 10 of 10 at 0.93-1.25 s (10 of 10,
    0.95-1.33); urge off: 60-81 deg in place (62-73), facing at 0.60-0.75 s; MDN 17.5-18.9 mm (17.9-19.7), HS 0.1 Hz and 25-28k
    events/s while backing (0-0.6 Hz, 26-30k); quiet arena HS 5.5, 10.1, 7.9, 76.5, 108.9 Hz (5.5, 10.1, 7.9, 72.6, 108.2: seeds 0-2
    **to the decimal**, 3-4 apart by the jump rule), brain 26.0, 30.6, 25.0, 94.9, 111.1k (25.9, 30.4, 24.9, 78.0, 72.3k), high
    states on seeds 3 and 4 from 3.8 and 5.15 s with no escape (3.9 and 6.2 s with 7 and 21 escape ticks); 868-874 MB (876). Its
    real-time factor 1.3-1.9 is the raw tick speed (the tool does not sleep to real time; the published 0.90-1.23 is the live loop's).
  - **physics (raw pose)**: speed 4.0 / 8.7 / 14.3 (4.1 / 8.9 / 14.9); turn 32 / 85 / 165 (28 / 86 / 174: +14 % at drive 0.3, within
    5 % above it); lure left 83, 103, 95, 101, 121 (102, 103, 105, 94, 94), right -101, -93, -97, -99, -120 (-87, -106, -100, -90,
    -125); faces 10 of 10 at 1.23-1.98 s (8 of 10 at 1.65-2.55); urge off: 0.2 deg, the lure still 70-71 deg off (0.4 deg, 69-71);
    MDN 19.0-21.4 mm, mean 20.0 (19.5-21.6, 20.4), heading drift 0-10 deg (3-10), HS 47-83 Hz and 98-123k events/s while backing
    (57-84 Hz, 68-103k); quiet HS 24.8, 17.7, 14.4, 18.4, 19.0 (24.8, 13.7, 14.4, 18.5, 19.0: four of five seeds), brain 73.4, 59.2,
    51.4, 48.2, 74.5k (56.2, 46.7, 43.1, 40.5, 58.8k), high states on seeds 0, 1, 2, 4 from 4.05, 7.9, 8.9, 7.35 s (seeds 0 and 2
    from 8.9 s); contacts per step 10-19; 1,570 MB (1.32-1.35 GB); real-time factor 0.137-0.187 under load 7-10 (MuJoCo alone
    0.149-0.212), 0.22 solo (Phase 0's bench: 0.216; published 0.086-0.113 on the machine the table was measured on).
  - **stride average**: lure left 100-103 (99, 90, 113, 101, 126), right -73 to -100 (-62 to -118); faces 9 of 10 at 1.68-2.10 s
    (6 of 10 at 1.83-2.83); urge off 0.4 deg (0.3); MDN 19.8-21.2 (19.2-21.4), HS while backing 0.0-24 Hz and 30-91k (0-0.4 Hz,
    27-31k); quiet HS 60.6, 9.7, 8.6, 6.8, 11.0 (57.7, 11.6, 8.8, 6.9, 9.0); high states on seeds 0, 1, 3, 4 (0 and 3).
- Decision 25's verdicts against the published numbers: drawn all within tolerance; physics speed within, turn outside at drive 0.3
  (+14 %), faces 10 against 8 (outside by 2); stride average faces 9 against 6. What the differences say: where nothing moves in the
  dish the runs **are** the published runs (the quiet arena to the decimal on 3 of 5 drawn seeds and 4 of 5 physics seeds); with a
  lure the exact protocol is lost (the wiggle is a reconstruction) and a spiking brain amplifies small differences, so the per-seed
  numbers scatter around the published ones with the same picture; the physics columns' events/s run a quarter higher than
  published at the same HS rates (the tool now records the spikes/graded split, so the next run can say whether the published
  number counted spikes alone); the stride-average column no longer removes HS while backing (0-24 Hz against 0-0.4). No code in
  `physics.py` or `vision.py` changed around the stride average since v2.8.0 (git diff), so the published stride-average runs
  differed in protocol too. **The gate for the levers is therefore this run** (same machine, same code), with decision 25's
  tolerances applied to it: put to the owner (Next step).
- Where a physics tick's time goes (`../runs/p4_profile_physics.py`, `../runs/p4-profile.log`; one fly, seed 0, parts on, 3 s after
  20 ticks, solo): real-time factor **0.220**; MuJoCo (Walker.advance: the CPG, the control writes and mj_step) 12.7 of 13.6 s =
  **93 %** of the wall time; **0.359 ms per mj_step** (89.8 ms per 250-step tick), the Python around it 0.027 ms per step (3.1 ms per
  tick of CPG and writes: 7 % of the step); the brain 11.6 ms per tick; senses 0.6 ms; contacts per step 10-20 while walking. So
  the levers must act on mj_step (the solver, the contacts, flygym's duplicated self pairs); vectorising the Python buys at most 7 %.

### Phase 4: the pair world (plan 8.3-8.5; 2026-10-03; **new** `virtual_fly/physics_pair.py`, `tests/test_physics_pair.py`; SCIENCE.md 13)
- Built as the design in the session scratchpad says (`p4-design.md`): flygym's multi-fly `Simulation` on the kit's walled floor
  centred at the origin; each fly spawned at its kit pose and then moved by its root free joint so that the thorax lands on the pose
  exactly (the 0.496 mm thorax offset measured on the built model); `place()` moves a fly without a rebuild; one CPG per fly
  (seed + 1000 k); one `mj_step` per 0.1 ms for both; the game sets both drives, the world steps once per tick, each body reads its
  thorax, its own wall contact and its tap. Decision 22's contact set: 233 distinct pairs, never Tarsus5. `physics.py` untouched.
- Checked by hand and by the seven tests: placement to 1e-6 mm; contact within 0.25 s from 3 mm (0.4 s from 6 mm) head-on; the tap
  with a force (2-18 units); backing away separates at once; each fly feels only its own wall; deterministic bit for bit; a two-fly
  physics game ticks, reports `physics.pair`, `physics.tap` and `scale` 0.389, and closes; the install hint without flygym.
- **Flies walking head-on slide past each other**, deflected by their rounded heads (the first contact's normal points mostly sideways;
  fly 0 veers 0.9 mm and 9 deg and passes along the other's flank). Found while chasing what looked like pass-through: a static probe
  with the thoraxes overlapping finds the thorax-thorax contact (penetration 0.49-0.94 mm), so the collider is sound; the dynamic
  configuration was simply beside, not inside. Tried against it, not adopted: stiffer pair `solref` (no change), convex hulls (a
  contype bit: +8 % walking speed, 1.3x faster: a lever for 8.6), MuJoCo's native CCD (no change). Options kept on `PairWorld` for 8.6.
- Speed (`tools/bench_two_flies.py --only pair-physics --contact-sets forelegs,full,none`, parts on, brains in processes, 80 ticks;
  `../runs/p4-bench-pair-physics.json`, load 0.8): forelegs **rtf 0.085** (tick p50 234 ms, p99 1,113 ms: the ticks in contact, where
  flygym's Newton solver at 1,000 iterations and 1e-12 converges slowly), MuJoCo 95 %, 1.11 ms per step for both flies, 24 contacts per
  step; full (1,521 pairs) 0.055, p99 1,792 ms, 1.74 ms per step; none 0.110, p99 292 ms, 0.85 ms per step. One fly alone: 0.22
  (0.359 ms per step). The game process 2.6 GB, the brain children 0.5 and 0.7 GB. The research's two-fly default was 0.037 on a
  loaded 4-core machine.
- `Ctrl+C` on `fly_game.py --partner female --body physics` (the 1.7 recipe, `../runs/p4-ctrlc-pair.log`): both flies in the state
  (0.7 s simulated in 6 s wall: 0.12x with the brains in their own processes), "Bye!", **exit code 0**, nothing left (8.11).
- The game: `Game(partner=..., body="physics")` builds the shared world and hands each fly its body; `--stride-average` with a
  partner is refused (the pair's senses see each body as it is); `--partner-body` must agree with `--body` (mixed bodies are not
  this phase's). The senses: the tap from MuJoCo replaces the drawn 3.4 mm rule (channel 3), a physics other is seen as a 0.7 mm
  cylinder 1.1 mm tall (`vision.SEEN_FLY`), both flies drawn at real size in 2-D and 3-D (`fly.scale`), the page's pace line says
  so, `whats_real` carries the pair lines (one world, the physical tap, real size, the male in the female body model).

### Phase 4: record-and-replay on disk (plan 8.7-8.8; 2026-10-03; **new** `virtual_fly/recording.py`, `tests/test_recording.py`; SCIENCE.md 13.4)
- The format as the plan says it (`recordings/<stamp>/`: header.json, frames.jsonl.gz, qpos.npy, t.npy, model/ with the exported
  MuJoCo model and its assets, poses.f32 at the stop); the `capture` action (refused with the reason: one at a time, an unwritable
  folder, the 2 GB cap; a recording in progress stops at the cap or when a single fly's world is rebuilt); `state.capture`; the
  replay endpoints (`/api/replays`, `/api/replay/<id>/header|frames|poses`) without the wildcard CORS header, the id a plain
  folder name; the capture action taken only from the server's own page (foreign Origin or Host: 403); `allow_pickle=False`.
- Checked (six tests): a drawn run saved and listed; the cap and an unwritable folder refuse with the folder named; no CORS header
  on the replay endpoints while the live ones keep theirs; foreign Origin and Host refused, the own page's three spellings
  accepted, a non-writing action not origin-checked; a physics-pair recording reloads its exported model and reproduces every
  geom's position **exactly** (0.0 mm; the live world's kinematics refreshed first, since after a step MuJoCo's positions are one
  step stale), and its poses agree with the 3-D view's gait atlas convention (the standing thorax entry within 1e-3 mm).
- Rendering under WSL2 (plan 8.9's **verify first**): `MUJOCO_GL=egl`, `osmesa` and `glfw` all render here (a 640 x 480 frame in
  6 ms each; libEGL, libOSMesa and WSLg's display are present); `egl` is the choice for the video tool.
- Gotcha: a script that builds a game with brain processes must guard its entry point (`if __name__ == "__main__"`): the spawned
  children import the script, and an unguarded one builds a game inside each child (the first sample run died that way).

### Phase 4: the speed levers as switches (plan 8.6; 2026-10-03; code only, the measurements come last by the owner's order)
- `physics.LEVERS`: dedupe (flygym's duplicated self pairs: 1,086 of 2,172 dropped per fly, exactly), solver100 (iterations 1,000 →
  100, tolerance 1e-12 → 1e-8), noslip5, noslip0, noself, simple (seqik_simple with tarsi-only floor contacts), dt2 (0.2 ms). Each is
  off unless asked for: `fly_game.py --physics-levers LIST`, `Game(physics_levers=...)`, `PairWorld(levers=...)`, `make_body(levers=)`,
  `tools/physics_table.py --levers LIST --baseline ../runs/p4-table.json` (the comparison against the tool's own run, decision 25's
  verdicts relative to it) and `tools/bench_two_flies.py --levers LIST`. With no lever nothing changes (the 18 real golden hashes
  checked again with the code in place: `../runs/p4-golden2.log`).
- First numbers, one body alone at full drive for 3 s (not the table yet): `simple` walks 15.4 mm/s (14.2 without) at rtf 0.32
  (0.22); `simple,dedupe,solver100,noslip0` 15.3 mm/s at rtf 0.55. The script for the whole measurement is
  `../runs/p4-levers.sh` (ten lever sets through the table against the baseline, then the pair bench per set; about 45 min idle).

### Phase 4: the replay player and the video (plan 8.8, 8.9; 2026-10-03; two forks on the sample recording `recordings/20261003-210246`)
- The page (`index.html`, `style.css`, `app.js`, `panels.js`, `arena3d.js`): a "Save replay" switch and the saved-replays list in the
  Recording card; a player bar in the stage (play/pause, 0.1-4x, scrub, the time, Leave) fed from the page's own animation loop at
  the recording's tick times the speed (smooth at 1x whatever the run's own pace); the live ticks kept behind the replay and shown
  again on Leave (the reload guard untouched); `?replay=<id>`; the badge "replay of a recorded run" over both views; the 3-D view
  moves every leg joint from the recorded poses (fly index = fly id) with the hand-built wing, abdomen and proboscis rotations on
  top, and says so on its badge. Limits accepted: a frame has no internal state, learning, lab lists, events or retina images (those
  panels show neutral values); frames are held whole in memory (an hour is about 150k lines); the pointer still acts on the live
  game. Tests: `test_web.py` 17, `test_browser.py` 7 (a drawn replay played, scrubbed and left; a physics-pair replay driving the
  legs from the recording) = 24 passed in 27 s locally (the LD_LIBRARY_PATH workaround for Chromium).
- The video (`tools/render_replay.py`, `tests/test_render_replay.py`): the recording's model and qpos replayed through MuJoCo's
  kinematics at the video's frame rate (joints interpolated, root quaternions normalised), cameras overhead / follow / side with
  the aim smoothed over 0.25 s, the second fly tinted lighter, a label; MP4 by OpenCV (mp4v). **Measured**: the 8 s sample at
  1920 x 1080, 30 fps: 241 frames in 40 s overhead (3.1 MB) and 36 s follow (3.6 MB), 6-7 frames rendered per second, about 130 ms
  per frame at any size (the exported model's 1,005,562 mesh triangles, not the pixels); `MUJOCO_GL=egl` is the default and works
  under WSL2 (osmesa and glfw too). 8.11's criterion met. The render test needs `MUJOCO_GL=egl` exported before pytest (physics.py
  alone sets disable): `../runs/p1-verify.sh` exports it now.
- A look at both: the headless screenshots in the session scratchpad (`shot-replay-2d.png`, `shot-replay-3d.png`) and the check
  frames (`p4-check*/check-middle.png`): the 3-D replay and the video show both flies with NeuroMechFly's meshes at real size, the
  female lighter, the legs as MuJoCo moved them.
- Found on the way: the new `capture` state key changed every golden frame's bytes (the hashes cover the whole state), so the key
  is sent only while a replay is being saved; the 18 real hashes are unchanged again (`../runs/p4-golden3.log`).

### Phase 4: the speed levers measured (plan 8.6; 2026-10-03; `../runs/p4-levers.sh`, `../runs/p4-table-<set>.json`, `../runs/p4-bench-pair-<set>.json`; SCIENCE.md 13.5 holds the two tables)
- Ten lever sets through `tools/physics_table.py --bodies physics --levers SET --baseline ../runs/p4-table.json` (30 runs each, four
  at a time, 3-5 min each, load 3-6) and the pair bench per set. Within decision 25's tolerance: dedupe, solver100, noslip5,
  noslip0, noself, dedupe+solver100, dedupe+solver100+noslip5; outside: dt2 (speed +17 %, turn +25 % at drive 0.3), simple and
  simple+dedupe+solver100+noslip0 (turn -18 to -19 % at full drive).
- **dedupe is free**: every number the baseline's, the single physics fly's golden hash bit-identical with it on (checked directly:
  `b48a56e59f1b` both ways), the single fly 1.27x faster, the pair 0.087 → 0.100. Adopted as the pair's default on 2026-10-03 and,
  on the owner's word of 2026-10-04, as every physics body's (`physics.DEFAULT_LEVERS`, `--physics-levers none` to switch it off);
  the 18 real golden hashes checked again with it as the default (`../runs/p4-verify3.log`).
- The noslip levers keep the speeds and turns but raise HS in the quiet arena from 18.9 to 34-37 Hz (the body's wobble as the
  retina sees it); solver100 buys the pair nothing (0.084); noself 1.76x single, pair 0.126; the fastest in-tolerance set,
  dedupe+solver100+noslip5: single 1.86x, pair 0.112. Put to the owner with the table.

### Phase 5 opened: the real-time question answered by a flygym 2.x probe (plan 8.1, decision 20; 2026-10-04; `../venv312`, `../runs/p5_flygym2_probe.py`)
- The owner, after watching the live physics pair at 0.16x: "make sure it moves in real time". No lever reaches it (13.5; the best
  in-tolerance pair set 0.112). flygym 2.1.0 (PyPI, with MuJoCo 3.9.0) installed in a Python 3.12 venv outside the repository
  (`../venv312`, 2026-10-04): one NeuroMechFly on flat ground with position actuators on the leg joints and adhesion, as the kit
  drives its 1.2.1 body. **Measured, 1,000 steps after 300 settling: one fly standing rtf 1.91 (0.052 ms per 0.1 ms step), two flies
  standing 0.95 (0.105 ms per step), two flies churning their legs (a crude 12 Hz stepping pattern, contact-heavy) 0.99, one fly
  churning 1.91.** Against 1.2.1 here: one fly 0.22, two 0.087 (walking, in the game). Why: 2.1's model has 72 dofs and 70 geoms per
  fly with simplified meshes (2,000 faces at most), 55 contact pairs per fly (1.2.1: 2,220), and ships with the solver at 100
  iterations, 1e-8 tolerance and 5 noslip iterations (the settings my levers tried) at the same 0.1 ms step; adding a second fly
  needs `add_ground_contact_sensors=False` (the sensor names clash, as the research found).
- What a migration would mean (not started: plan 1.8 item 13, ask first): the physics extra moves to flygym 2.1 and MuJoCo 3.9
  (Python 3.12 or newer; 3.10 and 3.11 lose the physics body), a new walker on the 2.x API (no CPG or recorded-step helpers in 2.x:
  the kit carries its own stride, already in the gait atlas), the pair world, the contact pairs, the tap and the recording's export
  ported, and the whole 6.7 table measured again on the new body (the gate), since its gait is not 1.2.1's. With the brains in their
  own processes the two-fly game should land near 0.9-1.0x real time on this laptop; a real gait costs more than churning, so that
  is an estimate until measured.

### Phase 5: the BANC file built (plan 9.2-9.4; 2026-10-04; **new** `tools/pin_banc.py`, `virtual_fly/banc.py`, `virtual_fly/specs.py`; `flywire.write_flyb` parametrised)
- The deposit's listing read from Dataverse (no key, a User-Agent): four published versions; version 3.0 ("Publication version",
  2026-07-01) pinned by id, size, MD5 and SHA-256 (decision 28, flagged: the v3 edge list and the transmitter table are
  byte-identical in every version, only the metadata table was revised). Downloaded in 5 min (0.44 GB) to `data/banc-src`.
- The build: **155,704 neurons, 9,599,814 connections, 30,011,413 synapses, 17 s, 2.4 GB peak** (the plan's estimates: a minute,
  1.9-2.6 GB). Dropped, every count in the meta: 13,064 glia/trachea/non-neurons, 19,595 unproofread rows, 145 duplicate ids;
  4,021,051 edges (12.3 M synapses, 29 % of all) whose end is a dropped id, 3,856,620 of them to one of **15,917 ids absent from
  the metadata table** (unproofread segments the v3 edge list keeps; FlyWire's source table carried only the published model's
  proofread neurons, so its build kept just 14 unannotated ids: the default of decision 30 therefore matches FlyWire's effective
  rule, and "keep every connected id" stays the switch). 11,889 kept neurons have no super class in BANC (10,777 optic lobe, 963
  nerve cord, 78 central brain): left blank, counted. The class, subclass and nerve maps by majority vote over the 23,608 neurons
  BANC matches to the male file: kenyon_cell → Kenyon_Cell (0.995 of 388 votes), antennal_lobe_local_neuron → ALLN (0.96),
  mushroom_body_dopaminergic_neuron → DAN (1.0), mushroom_body_output_neuron → MBON (0.89), olfactory_receptor_neuron → olfactory
  (0.996), bristle_neuron → mechanosensory_tactile (0.455), chordotonal and campaniform → mechanosensory_proprioceptive; the
  nerves to the kit's abbreviations (left/right_antennal_nerve → AN at 0.99, anterior_dorsal_mesothoracic → ADMN at 0.93, the leg
  nerves → ProLN/MesoLN/MetaLN at 0.75-0.89, ...); multi-nerve values blank. On the file: `class:Kenyon_Cell` 4,438, `class:ALLN`
  427, `class:DAN` 299, `class:MBON` 104, `class:olfactory` 2,832, `nerve:ADMN` 959, `subclass:grooming` 217, `subclass:wind_gravity`
  0 (the audit will say what carries it). Transmitters: 79k acetylcholine, 20.5k glutamate, 19.4k GABA, 5.9k histamine, 5.8k
  dopamine, 737 octopamine, 769 serotonin, **127 tyramine** (sign +1, decision 31), 23.4k unclear. The kit's names BANC spells the
  same: MN9, DNp01, pC1a-e, KC types, JO-A/B/C/E, LgLG1a/1b (the leg taste cells), DLM1-4/DLM5; by hand: GNG232 → CB0616,
  GNG087 → CB0219, pC1_ → pC1, VS, KCa'b'; absent (for the audit): TTMn, ps1, hg, AN19A018, AN_SMP_2, AVLP568. The `malecns_cell_type`
  column names none of the kit's missing names, so no data-derived alias came out.
- Somas: `root_position_nm`; the brain at y 39-600 k nm, the cord down to y 1,107 k: the long axis is y (`layout_axis_hint: "y"`).
- The FlyWire writer gained `dataset` and `nt_sign` parameters and `np.searchsorted` for the edges' rows: the female rebuilt in
  14 s and compared with the kit's file: every byte after the meta block identical (97,453,277 bytes), the tables and the meta
  identical but the build time (`../runs/p5_flywire_identity.py`).
- **The gain prior (9.7)**: median input synapses per neuron, cb_intrinsic: male 730, FlyWire 332, BANC 147 as shipped (204 with
  every connected id kept; the research's 208). Priors: 332/147 = **2.26** (shipped), 332/204 = 1.63 (every id); all neurons
  200/93 = 2.15. The sweep (`../runs/p5-gain-sweep.sh`) takes 1.0, 1.5, 1.63, 2.0, 2.26, 2.85 on the shipped file once `--fly
  banc` exists; the rule is written at the top of the script before any result is looked at.

### Phase 5: the kit's names on her cells (plan 9.5; 2026-10-04; **new** `tools/alias_audit.py` by a fork; `../runs/p5-alias-audit.md`)
- The audit (153 specs from `virtual_fly/specs.py`, each term counted on the three files, candidates for every zero): male 147 of 153
  specs whole (the empty ones the female readouts), FlyWire 133, BANC 125 at first. Fixed by hand (AN19A018 → prefix:AN19A018,
  regex:^DLMn → DLM1-4,DLM5, regex:^hg → iv1-iv4, subclass:wind_gravity → JO-C,JO-E) and by the data-derived aliases widened to the
  MANC and FAFB cross-match names and to regex terms (TTMn 2 cells, AN_SMP_2 1, regex:^ps1 3): **BANC 141 of 153**, the 11 left
  explained (pIP10, the male's song neuron; gene:fru/dsx, decision 32; AVLP568, not in BANC). The collector's one false positive
  (olfaction's odour keys) fixed. BUILD 2.
- Checked on the file: LgLG1a,LgLG1b 304 (the leg taste cells the male's touch can now reach), prefix:JO-C/L,prefix:JO-E/L 230,
  subclass:wind_gravity 407, nerve:ADMN 959, class:Kenyon_Cell 4,438.

### Phase 5: the gain, the comparisons, the checks (plan 9.7-9.9; 2026-10-04; `../runs/p5-sweep-*.json`, `../runs/p5-pair-banc.json`, `../runs/p5-verify1.log`)
- The sweep (the rule written first, SCIENCE.md 14.3): no gain passes the classic six (best 3 of 6; every gain above 1.0 brings
  runaways: 10-11 of 12 experiments in the game profile at 2.0 and above); what fails (sugar → G2N-1/MN9 0 Hz, dust → aDN 0 Hz,
  looming → giant fibre 120-180 Hz) does not move with the gain; **the default gain stays 1.0** (2 of 6 classic, as FlyWire's
  female; no runaway). The whole sweep took 2 min (12 runs, three at a time).
- Comparisons (SCIENCE.md 14.4): classic 2/6 (FlyWire 2/6, male 6/6); the song motor neurons readable at last and 0 Hz (no pIP10);
  the parts list 7,478 modulatory neurons (FlyWire 2,318: the dopamine over-calling triples the tones); the pair experiments 7 of
  14 readouts, her giant fibre 99 Hz under the song (FlyWire 7-20): the kit's song level would startle her.
- `fly_game.py --partner banc`: both brains in processes, Ctrl+C → "Bye!", exit 0, nothing left (`../runs/p5-ctrlc-partner-banc.log`).
- Verification: **649 passed, 7 skipped** (the browser tests, run apart in Phase 4), the 18 real golden hashes unchanged
  (`../runs/p5-verify1.log`); draft PR #21 opened (`60cd766` and after).

### Golden hashes (plan 4.9)
- Synthetic (`tests/golden_single_fly.json`): nine configurations, made with Python 3.12.3, NumPy 2.5.3, numba 0.67.0; a second
  run reproduces every hash (the test passes in normal mode; a determinism test runs one configuration twice).
- Real data (`../runs/p0-golden-real.json`, not committed): `tools/golden_hashes.py --save` wrote 18 hashes (male and female,
  the same nine configurations; 400 ticks took 2.5-4.5 s each with the walking urge, 1.0-1.3 s without it, the 80 physics
  ticks about 10 s), same versions. Rerun with
  `--compare ../runs/p0-golden-real.json` at the end of every phase.

## Open issues
- **The two-fly physics game is at half real time, not real time** (v3.0, 2026-10-04): one physics fly runs at 1.04x, the pair at
  0.49 (0.62 in the live game of the Ctrl+C check; 0.69 with `--physics-levers dt2`). MuJoCo and the controller are 69 % of the
  pair's tick; the other 16 ms are the game's own two-fly work (senses, brain traffic, the state), which no body setting touches.
  The route left: let the brains compute while the bodies step (one tick of sensorimotor latency, a game-loop change; real flies
  have 20-40 ms visuomotor latencies), which is the owner's call, or a multithreaded solver MuJoCo's Python bindings do not expose.
- **The physics golden hashes changed** (by nature: a new engine and body): `tests/test_golden_single_fly.py[physics_body]` fails
  locally and `tools/golden_hashes.py --compare ../runs/p0-golden-real.json` reports the two physics configurations changed (16
  unchanged). Re-save on the owner's word: `VF_UPDATE_GOLDEN=1 .venv/bin/python -m pytest -q tests/test_golden_single_fly.py` and
  `.venv/bin/python tools/golden_hashes.py --save ../runs/p0-golden-real.json` (the synthetic file is committed, the real one not).
- **The turn rate at full steering is 129 deg/s, not 165** (SCIENCE.md 6.7.1): the 1.2.1 body itself gives 136 under MuJoCo 3.9 with
  the kit's controller, so this is the engine's solver, not the port; speeds match to 1 %. The lure is faced on 9 of 10 runs (10 of 10
  before; the published 1.2.1 table said 8 of 10).
- **The owner's view of the physics body (2026-09-27, after looking at the three games):** the drawn-body games look fine; the
  physics body "is less realistic than the other one and barely moves and jumps properly". The owner wants both bodies to stay
  selectable (the physics body enabled or disabled) and hopes the physics mode performs better by the end. For Phase 4: keep the
  drawn body the default in every mode and the physics pair opt-in (`--body physics`, as the plan already says); put the record-
  and-replay at 1x and the offline video first among its deliverables, since live physics stays slower than real time by nature;
  and note that NeuroMechFly has no jump (a giant-fibre burst is an "escape command" with the legs standing), so a physical jump
  would be a new hand-built controller: ask the owner before adding one.

## Next step
The owner reads the draft PR of the migration (the numbers in SCIENCE.md 6.7.1 and below) and says: whether v3.0.0 is the version
they meant ("version 3.0"), whether the physics golden hashes may be re-saved (Open issues), whether `dt2` stays a switch, and
whether the pair's remaining half (the game loop's own work) is worth a latency-for-speed change of the game loop. Then: re-save
the hashes, mark the PR ready, merge on their word. The MuJoCo Warp question (plan 8.10) stays closed as before.

## Session notes
### 2026-10-04 (the flygym 2.1 migration: v3.0.0, a draft PR)
- **Why the first fit walked at 2 mm/s sideways:** the 3-D view's gait atlas is a whole-body tripod cycle (frame k = leg LF at
  phase k, the legs RF, LM and RH half a cycle on), and the first refit read it as a per-leg table. Found by rebuilding each
  leg's foot path from the fitted angles (three legs moved forwards in "stance"); fixed by rolling the second tripod's frames by
  32 (`tools/refit_stride.py`, the shipped `web/models/nmf_stride.npz`, 37 KB: the angles, the swing windows, every atlas geom's
  fixed offset on the 2.1 bodies, the 1.2.1 inputs the next run reads back; `--check` rebuilds the atlas to 0.003 mm and 0.4 deg).
- **Why flygym 2.1's own joints were not kept:** stiffness 10 and damping 0.5 fight kp 45 (tracking error 9 deg, 10.3 mm/s);
  with 1.2.1's `Fly` defaults (0.05/0.06, tarsi 7.5/0.01, kp 45, ±65, adhesion 40) the body walks at 14.3 mm/s, straight. flygym
  2.1's `MjsJoint.stiffness` is a 3-vector in the bindings (set `[k, 0, 0]`). Its `Fly` alias warns; `NeuroMechFly` is used.
- **Units:** flygym 2.1's masses are grams, forces micronewtons (the fly weighs 10.05 µN); 1.2.1's exported model has the same
  mass (1.03 mg) and the same split (0.9 in the thorax group, 0.1 in the legs), so no gain was rescaled: the (wrong) ÷1000
  guess collapsed the fly (measured, discarded).
- **Written:** `virtual_fly/physics.py` on flygym 2.1 (`build_fly`, `add_wall`, `add_wall_pairs`, `FlyRig`, `CPG`, `Stride`,
  `Walker`, `PhysicsBody` unchanged in interface; the levers' 2.1 semantics; `DEFAULT_LEVERS = ()`), `physics_pair.py` on it
  (the part names stay 1.2.1's, `physics.part_geom` maps them; two flies with `add_ground_contact_sensors=False`), `recording.py`
  (the model as a gzipped `.mjb`, `load_model` reads it and the old MJCF export; `PoseMapper` places the 1.2.1 geoms on the 2.1
  bodies), `tools/render_replay.py` (the thorax by either name), `tools/refit_stride.py` (new), `tools/build_fly_model.py`
  (the build needs 1.2.1: says so; `--check` works), the tests (`test_physics.py`, `test_physics_pair.py`, `test_recording.py`,
  `test_fly_model.py`: the stride check), the physics extra (`flygym==2.1.0`, `opencv-python-headless`), version 3.0.0 (the
  engine, the recordings' model and the install all change: a major bump; the owner's "version 3.0" was read as the BANC
  deposit's version, so this is flagged in the PR), README, SCIENCE 6.7 and the new 6.7.1, 13.1, 13.4, 13.5, 15, API,
  ARCHITECTURE, the NeuroMechFly NOTICE, the page's and the launcher's pace texts, "What's real here?".
- **Decided alone, flagged:** the wall's sliding friction 0.3 (at flygym's 1.0 the body climbs the 3 mm wall and falls on its side,
  at 0 the solver blows up; four approaches measured); the controller every 0.5 ms instead of every 0.1 ms step (speed, turn and
  wobble within 1 %; the pair 23 % faster); `dt2` left a switch (within tolerance on speeds and turns, the lure faced on 7 of 10
  against 9: outside decision 25); the version number.
- **Measured** (SCIENCE.md 6.7.1; `../runs/p6-table-k5.json`, `p6-table-dt2.json`, `p6-bench-k5.json`, `p6-bench-dt2.json`):
  speeds 4.0 / 8.7 / 14.5 mm/s (baseline 4.0 / 8.7 / 14.3), turns 45 / 86 / 129 deg/s (32 / 85 / 165), the lure faced 9 of 10
  (10 of 10), MDN 21.6 mm (20.0), quiet-arena HS 6-39 Hz (14-25), the one-fly game at 1.04x real time (0.14-0.19), the pair at 0.49
  (0.085; `full` 0.46, `none` 0.52; `dt2` 0.69), the live game 1.00 and 0.62 (the Ctrl+C checks, both exit 0 with nothing left),
  peak memory 1.07-1.14 GB (1.57). An 8 s pair recording (`recordings/20261004-203235`, 47 MB with the gzipped model) renders to
  a 720p MP4 at 23 frames per second (6-7 before). The wobble question settled by running the 1.2.1 body from the sample
  recording's exported model under MuJoCo 3.9 with the kit's controller: 260 against 256 deg/s, 136 against 130 deg/s turning,
  15.5 against 14.3 mm/s: the two bodies agree; the baseline's 165 was MuJoCo 3.2.7. Ruled out with measurements: 1.2.1's
  contacts, noslip 100 with the Newton solver at 1,000 and 1e-12, the elliptic cone, the mass split, self-collisions (none
  changes the wobble), MuJoCo threads and islands (not in these bindings), the wall as bitmask collisions (5 %).
- **Verified:** `../runs/p1-verify.sh`: 649 passed, 1 failed (the physics golden hash, expected), 7 browser tests skipped there and
  run separately (7 passed, the physics replay in the 3-D view included); the golden compare 16 unchanged, 2 changed (both
  physics); `tools/refit_stride.py --check` ok; `tools/build_fly_model.py --check` ok.
- Helper scripts in `../runs/`: `p6_walk_check.py`, `p6_walk_diag.py` (the kinematic stroke check that found the tripod
  mix-up), `p6_contact_check.py`, `p6_calib_check.py`, `p6_wobble.py`, `p6_wobble2.py`, `p6_old_body.py`, `p6_old_body2.py`
  (the 1.2.1 body under MuJoCo 3.9), `p6_mass.py`, `p6_selfcoll.py`, `p6_speed.py`, `p6_speed2.py`, `p6_dt2.py`,
  `p6_wallcost.py`; the exports `nmf_step_flygym121.npz`, `nmf_geoms_flygym121.npz`, `cpg_controller_121.py`, the refits
  `nmf_stride_v2.npz` (atlas order) and `v3` (per leg); the flygym 1.2.1 wheel unpacked in the session scratchpad for reading.

### 2026-10-04 (Phase 5 merged, the migration opened)
- The owner: "sure gain 1, mark pr and merge, route to real time, version 3.0" → PR #21 marked ready and squash-merged as
  `ff16d8c` (v2.13.0 on `main`). Branch `claude/flygym2-realtime` from `origin/main`; flygym 1.2.1's stepping data, CPG source and
  geom offsets exported from the kit's venv before the engine is replaced.

### 2026-10-04 (Phase 5 opened)
- The owner: "merge" → PR #20 squash-merged as `5a27efe` (v2.12.0 on `main`). Then, after seeing the live physics pair at 0.16x
  real time: "bruh its so fckin slow... do phase 5 and make sure it moves in real time". The game stopped; branch
  `claude/two-flies-p5-banc` from `origin/main`; flygym 2.1.0 installed in `../venv312` for the real-time question (plan 8.1,
  decision 20: a spike outside the repository, only if the user asks); the BANC deposit's listing read from Dataverse
  (four published versions; version 3.0 pinned, flagged) and the pinning tool written.
- The build (17 s), the vote maps, the aliases (an audit fork), the loader plumbing and the data-vs-sex fixes (a fork), the
  gain sweep (no gain meets the rule), the pair experiments, the parts count, the full verification (649 passed, 18 hashes
  unchanged); SCIENCE.md 14, README, API, ARCHITECTURE; v2.13.0; draft PR #21.

### 2026-10-04 (Phase 4, the end)
- The owner: "Go with the recommendations" on the five questions of 2026-10-03: the gate accepted, `dedupe` the default for every
  physics body (bit-identical golden frames), the faster preset declined, the head-on sliding accepted, Chromium's libraries left
  to them. The default applied to the game, the CLI, the tools and the tests; the full verification with it; PR #20 marked ready.

### 2026-10-03 (Phase 4 opened)
- The owner read the 3-D badge: "ok yeah the fps is at 80 or so, quite good. go ahead and merge". The number recorded (SCIENCE.md 12,
  Measurements, the PR body; `ca8723b`); PR #19 squash-merged as `157ffa9` on that word (v2.11.0 on `main`); the demo game on port 8765
  stopped with the 1.7 stop block (nothing left in its session).
- Branch `claude/two-flies-p4-physics-pair` from the fetched `origin/main`; decisions 20-26 put to the owner with the defaults, the
  physics-body issue and the ordering question. The owner: "Defaults" (all seven at the plan's defaults, the levers last, no
  hand-built jump); recorded in Decisions. 8.2 started.
- 8.2: `tools/physics_table.py` written, smoke-tested, run twice (a still lure, then the wiggling lure that recovers the
  published turn-in-place); the profile; the results and the gate question recorded (Measurements; `84b1f05`).
- 8.3-8.5: `physics_pair.py` designed (`p4-design.md` in the session scratchpad) and built; flies walking head-on slide past each
  other, which cost an hour of probes before the static overlap test showed the collider sound; the game integration, the
  senses at real size, the page scale and pace line, the CLI, seven tests, the pair-physics bench kind, SCIENCE.md 13, README,
  ARCHITECTURE, API; version 2.12.0; the Ctrl+C check (exit 0).
- Pushed; draft PR #20 opened. 8.7-8.8: `recording.py`, the capture action, the replay endpoints, six tests (`002aea3`);
  SCIENCE.md 13.4. The page's player and the video tool (8.9) as two forks on a sample recording.
- The forks' work in (`1948c77` the video, `c52bac0` the player); the `capture` state key made conditional after it changed every
  golden frame; the lever switches (`34be6d2`) and their measurement (ten sets, 50 min); dedupe adopted for the pair; SCIENCE.md
  13.5; the full verification and the single physics game's Ctrl+C check; PR #20's body refreshed.

### 2026-10-03 (Phase 3 started)
- PR #18 squash-merged as `63681db` on the owner's word ("merge and go with defaults"); branch `claude/two-flies-p3-3d-view` from the
  fetched `origin/main`. Decisions 18-19 recorded with their defaults.
- three.js vendored with its hashes (`119556a`), the licence verified, the installs, the packaging and the server's content types;
  version 2.11.0, the `browser` extra, SCIENCE.md section 12 and the architecture lines (`79f022c`); two forks built the model
  files with `tools/build_fly_model.py` (`6c079a9`: 58,293 triangles, 1.4 MB, the gait atlas) and the 3-D dish in the page; the
  full suite and the golden hashes checked on the whole tree; pushed; draft PR #19 opened.
- The owner: "go ahead and run chromium, and 3d view looks decent, would be good to add some coloring it is all brown": Chromium
  downloaded, the five browser tests passed (Measurements), the flies coloured by part; PR #19 marked ready.

### 2026-10-01 (Phase 2 started)
- PR #17 marked ready on the owner's word ("you merge, not me, and continue"); the automatic review posted nothing on the ready PR
  either; CI green; squash-merged as `f70d8be`. Branch `claude/two-flies-p2-gpu` from the fetched `origin/main`.
- Decisions 16-17 recorded with their defaults (put to the owner with the defaults; the owner's answer: "continue").
- CuPy installed and checked (Measurements); pytest and the real golden compare rerun after the install; a read-only map of the brain
  code (the NumPy and numba steps, the host-side blocks, construction, the tests, BrainIO and the CLI) written for the kernels (scratch,
  not committed), then a design document for the engine (scratch) from it and from the code itself.
- Step one (`8c5ec16`, `dd9fad3`): `FlyBrain(conn, backend="cupy")` builds `gpubrain.GpuFlyBrain` (chosen in `__new__`; "auto" never
  picks the GPU; "cuda" still refused); `advance_steps(n)` is the one stepping loop, `advance(n)` and `run(ms)` build on it and the
  seam's tick calls `advance(50)` (the 18 real golden hashes unchanged); `--backend cupy` in both programs with the one-line refusal
  before loading. Version 2.10.0 (`4cc1509`); SCIENCE.md section 11 placed (the arithmetic table; measurements to come), Honest
  limitations and References renumbered to 12 and 13; README's install text by driver generation; ARCHITECTURE.md (`5368adf`).
- The brain server of 6.5 (`f71842c`, one child process for every brain, `brain_procs="server"`, chosen by `auto` for a GPU brain;
  the re-test and rebuilds on the CPU) built and tested with CPU brains on the synthetic connectome (17 tests) while the engine was
  being built. `tools/bench_two_flies.py --backend` and `tools/bench_gpu.py` (`4c297ee`, `0420d3c`): the GPU's environment row
  measured a 225 GB/s device-to-device copy (36 multiprocessors, a 128-bit bus).
- The engine (`b7db401`): bit-identical to NumPy and numba on every test configuration and, in four real-data runs of 2 s, on
  every step and every state byte; the 16 experiments with `--backend cupy` identical to the baseline. Its speed work (`5ed284f`):
  a stream race fixed, six kernels per step, a warp per hit target, one upload and one download per chunk, the host's
  bookkeeping per chunk; then the two brains overlapped in the server (Measurements): the two-fly game at 3.6x real time on the
  GPU against 2.8x on the CPU. SCIENCE.md 11, README, ARCHITECTURE.md written with the measured numbers.
- Pushed; draft PR #18 opened; CI green on 3.10-3.12 (570 passed, 42 skipped without CuPy); marked ready; the owner asked whether to merge.

### 2026-10-01 (Phase 1, the end)
- The previous session ended on its usage limit with the third review launched but not run (its five agents failed at once) and the
  branch unpushed at `c957f5a`; this session started from the log's next step: 531 tests and the 18 golden hashes re-checked first.
- The third review rerun with seven lenses (Measurements); three forks fixed its points in parallel on disjoint files (the server
  side, the page, the documents and tools): `8c5ec16`, `41238fa`, `c9813b6`; 550 tests; 18 real golden hashes unchanged.
- The pair-game benchmark rerun on the idle machine (10.2 states both runs' loads); the three `Ctrl+C` checks on the final code; the
  90 s pair measurement rerun with the committed tool (Measurements).
- Pushed; draft PR #17 opened; CI green on Python 3.10, 3.11 and 3.12 (547 passed, 3 skipped without flygym); the automatic review
  posted nothing on the draft (it runs again when the PR is marked ready); the owner asked to look at `fly_game.py --partner female`
  in a browser.

### 2026-09-27 (Phase 1)
- PR #16 merged by Claude on the owner's go-ahead (squash). Branch `claude/two-flies-p1-two-brains` from `origin/main`.
- The owner answered decisions 4-15 with the defaults ("defaults"); recorded above.
- Steps 1-2 done (`89d6f78`, `ae3d656`), then an adversarial review of the seam (four lenses, two refuters per finding) confirmed nine
  points, all fixed in `2903e39` with a regression test each: the lockstep barrier releases every brain's lock when one child dies;
  the paused loop survives a dead child; a killed child reports its real exit code; the tick that resets a runaway brain publishes
  post-reset numbers as v2.8.1 did; a rebuild that fails in the child leaves the game usable; a request that times out stops the
  child instead of desynchronising the pipe; the scenario measure asks for the cheap number; one tautological test assertion
  removed. 471 tests; 18 real golden hashes unchanged.
- Steps 3-5 done (`cb7e853`, `0b7aa1b`, `31e5897`; 496 tests; 18 real golden hashes unchanged; the real two-fly game runs both brains
  in their own processes and exits cleanly). SONG_MAX_HZ measured (`217b187`), the pair experiments (`468f757`), the pair-game
  benchmark row (`6008897`), SCIENCE.md section 10 begun (`7fb119f`, `d616063`, `19dabfe`), version 2.9.0 (`f539523`).
- An adversarial review of steps 3-5 (five lenses, two refuters per finding) confirmed ten small points (her decision-neuron
  monitors lost on a local-brain rebuild; the male's "court" check keyed on the scripted female; the cues display flag off by
  the list default; touch without collide a no-op; the abdominal gesture on a partner without song cells; a watch able to take
  one of her built-in keys; three test gaps). Fixed in `9fa090d` with a test each.
- Steps 6-7 done: `6d0d87b` (`?fly=k` on every per-fly endpoint, `"fly": k` on per-fly actions, the state's `flies` list, per-fly
  recording frames and events; docs/API.md) and `d5d268f` (the page: `posesOf`, the fly menu `focusSel`, two persistent brain maps,
  per-fly actions through `setActionFly`, the three-state female toggle, her cues). Step 8: the pair scenario `pair_courtship` and
  the pair checks (`0d56642`), `tools/pair_courtship.py` and its 20 runs (SCIENCE.md 10.5), the pair-game benchmark, the
  16 experiments identical to the baseline. 531 tests; 18 real golden hashes unchanged after every commit.
- A third adversarial review (steps 6-8, the tools and SCIENCE.md section 10) in progress; then push, the draft PR and CI.

### 2026-09-27
- Read the plan; machine facts gathered (4.1, 4.3); no missing system packages (4.4).
- Cloned at `739e791`; venv created; the dev, female and physics extras and flygym 1.2.1 (`--no-deps`) installed.
- Fresh-clone checks (4.6, both blocks) all as the plan expects; findings above and in the plan copy.
- Branch `claude/two-flies-p0-baseline` from `origin/main`; this log and `docs/TWO_FLIES_PLAN.md` created.
- pytest, the four baseline experiment runs, the female build, the three games and the `Ctrl+C` check done (measurements above).
- Written: `tools/bench_two_flies.py` (from the plan), `tools/golden_hashes.py` (holds the shared CONFIGS), `tools/compare_experiments.py`,
  `tests/test_golden_single_fly.py` + `tests/golden_single_fly.json`; ARCHITECTURE.md's module tree lists them.
- Benchmark run twice (`../runs/p0-bench-first-run.json`, then `../runs/p0-bench.json` with the per-row processes); real-data
  golden hashes saved; SCIENCE.md 9.6 written; pushed; draft PR #16 opened; CI green.
- The owner looked at the three games in a browser ("look fine"; the physics body's realism: see Open issues) and confirmed
  decisions 1-3; PR #16 marked ready for review.
- An adversarial review (four lenses, each finding checked by two refuters) confirmed 9 small points, all fixed: `--compare`
  now names saved hashes a run did not recompute (and fails on them unless the run was narrowed); the experiment comparer
  ignores the integrator's name in the settings (a `--backend cupy` run must compare clean in Phase 2); the benchmark refuses
  a bad `--only`, `--seconds` or `--json` folder before loading; the pair-cost sentence in SCIENCE.md 9.6 now states the
  measured range and the run-to-run spread.
