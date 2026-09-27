# Two flies: progress log

Plan: docs/TWO_FLIES_PLAN.md. Newest session notes first.

## Status
| phase | branch | PR | state | last update |
|---|---|---|---|---|
| 0 | claude/two-flies-p0-baseline | #16 | merged (squash, `3631770`) | 2026-09-27 |
| 1 | claude/two-flies-p1-two-brains | #? | in progress | 2026-09-27 |

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
  Not installed yet: cupy (Phase 2), playwright (Phase 3), trimesh (Phase 3).

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

## Measurements
All on 2026-09-27, this machine, `nice -n 10`, one job at a time, machine otherwise idle (load average under 1.5).

### Tests (plan 4.7)
- `python -m pytest -q` on the fresh clone: **438 passed, 0 skipped**, 24 s (the two flygym tests run: the physics extra is installed).
- With `tests/test_golden_single_fly.py` added: **449 passed**, 35 s.
- CI on PR #16 (ubuntu-latest, `[dev]` only, no flygym): Python 3.10 with NumPy 2.2.6 and numba 0.67.0: **446 passed,
  3 skipped** (the two physics tests and the golden physics configuration); 3.11 and 3.12 green too. So the synthetic golden
  hashes made here with NumPy 2.5.3 reproduce under NumPy 2.2.6 on Python 3.10: no version-dependent hash set is needed
  (plan 4.9 item 3).

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
Wiring spot checks (`--female --inputs`): pC1a → DNp37 382 synapses (26 % of its input), vpoEN → DNp37 169, CL313 → DNp13 678 and
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
Phase 1 step 1 (plan 5.2): move the per-fly state of `Game` into `FlyAgent` with behaviour unchanged (golden hashes
identical, all tests green), then step 2 (5.3, the `BrainIO` seam).

## Session notes
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
  one of her built-in keys; three test gaps). Fixes, then steps 6-7 (per-fly API and state, the page), in progress.

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
