# Two flies: progress log

Plan: docs/TWO_FLIES_PLAN.md. Newest session notes first.

## Status
| phase | branch | PR | state | last update |
|---|---|---|---|---|
| 0 | claude/two-flies-p0-baseline | #16 | merged (squash, `3631770`) | 2026-09-27 |
| 1 | claude/two-flies-p1-two-brains | #17 | merged (squash, `f70d8be`) | 2026-10-01 |
| 2 | claude/two-flies-p2-gpu | #18 | merged (squash, `63681db`) | 2026-10-03 |
| 3 | claude/two-flies-p3-3d-view | #? | in progress | 2026-10-03 |

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

### Golden hashes (plan 4.9)
- Synthetic (`tests/golden_single_fly.json`): nine configurations, made with Python 3.12.3, NumPy 2.5.3, numba 0.67.0; a second
  run reproduces every hash (the test passes in normal mode; a determinism test runs one configuration twice).
- Real data (`../runs/p0-golden-real.json`, not committed): `tools/golden_hashes.py --save` wrote 18 hashes (male and female,
  the same nine configurations; 400 ticks took 2.5-4.5 s each with the walking urge, 1.0-1.3 s without it, the 80 physics
  ticks about 10 s), same versions. Rerun with
  `--compare ../runs/p0-golden-real.json` at the end of every phase.

## Open issues
- **The owner's view of the physics body (2026-09-27, after looking at the three games):** the drawn-body games look fine; the
  physics body "is less realistic than the other one and barely moves and jumps properly". The owner wants both bodies to stay
  selectable (the physics body enabled or disabled) and hopes the physics mode performs better by the end. For Phase 4: keep the
  drawn body the default in every mode and the physics pair opt-in (`--body physics`, as the plan already says); put the record-
  and-replay at 1x and the offline video first among its deliverables, since live physics stays slower than real time by nature;
  and note that NeuroMechFly has no jump (a giant-fibre burst is an "escape command" with the legs standing), so a physical jump
  would be a new hand-built controller: ask the owner before adding one.

## Next step
Phase 3 (branch `claude/two-flies-p3-3d-view`, plan section 7): vendor three.js with its hashes (7.1) and the packaging changes;
verify the meshes' licence and build the GLB and the gait atlas with `tools/build_fly_model.py` (7.2, 7.3); the 3-D dish in the page
(7.4) with the "animation, not physics" badge and the `whats_real` lines; the tests (7.6: the browser tests need the owner's word for
the Chromium download, plan 1.6); the docs; the draft PR.

## Session notes
### 2026-10-03 (Phase 3 started)
- PR #18 squash-merged as `63681db` on the owner's word ("merge and go with defaults"); branch `claude/two-flies-p3-3d-view` from the
  fetched `origin/main`. Decisions 18-19 recorded with their defaults.

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
