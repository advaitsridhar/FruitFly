# Two flies: progress log

Plan: docs/TWO_FLIES_PLAN.md. Newest session notes first.

## Status
| phase | branch | PR | state | last update |
|---|---|---|---|---|
| 0 | claude/two-flies-p0-baseline | #16 | draft PR open, CI green | 2026-09-27 |

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
| 1 | operating system route | WSL2 (Ubuntu 24.04), Claude Code inside Ubuntu: the machine already matched the default | default, taken in an autonomous session; the owner may override | 2026-09-27 |
| 2 | Python version | the system python3, 3.12.3 (in the 3.10-3.12 range) | default | 2026-09-27 |
| 3 | continue before a PR is merged | wait for the merge before the next phase | default | 2026-09-27 |

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

### The games (plan 4.7; checked through the API; the owner's browser look is still pending)
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

- **The pair rows are the Phase 1 number:** two busy brains side by side run at 2.6-2.8x real time (parts off) and 1.5-1.6x
  (parts on) on this CPU, well above the 0.7x target of plan 3.3, before any pipe or lockstep cost.
- The plan's script measured memory with `ru_maxrss`, which never falls within a process and which Linux hands to spawned
  children, so the first run reported the parent's high-water mark on every pair row (853 MB). The committed script runs every
  row in its own child and reads `VmHWM`; the first run's timings (`../runs/p0-bench-first-run.json`) were close to the second's
  except the male parts-off brain row (0.203 there, 0.138 here): treat the first decimal of ms per step as the precision.
- Physics: 0.22x real time here (the plan's reference machine: 0.086-0.113), 112 ms per tick.

### Golden hashes (plan 4.9)
- Synthetic (`tests/golden_single_fly.json`): nine configurations, made with Python 3.12.3, NumPy 2.5.3, numba 0.67.0; a second
  run reproduces every hash (the test passes in normal mode; a determinism test runs one configuration twice).
- Real data (`../runs/p0-golden-real.json`, not committed): `tools/golden_hashes.py --save` wrote 18 hashes (male and female,
  the same nine configurations; 400 ticks took 2.5-4.5 s each, the 80 physics ticks about 10 s), same versions. Rerun with
  `--compare ../runs/p0-golden-real.json` at the end of every phase.

## Open issues
- The owner has not yet looked at the three games in a browser (Phase 0 acceptance criterion 6).
- The decisions of section 10 for Phase 0 were taken with their defaults in an autonomous session: the owner should confirm or
  change them when reviewing the PR.

## Next step
Phase 0: read PR #16's CI result and the automatic review (`gh pr checks 16`, `gh pr view 16 --comments`); fix what they raise;
when the acceptance criteria hold, mark the PR ready (`gh pr ready 16`) and wait for the owner. Then Phase 1 (plan 5): put
decisions 4-15 to the owner first.

## Session notes
### 2026-09-27
- Read the plan; machine facts gathered (4.1, 4.3); no missing system packages (4.4).
- Cloned at `739e791`; venv created; the dev, female and physics extras and flygym 1.2.1 (`--no-deps`) installed.
- Fresh-clone checks (4.6, both blocks) all as the plan expects; findings above and in the plan copy.
- Branch `claude/two-flies-p0-baseline` from `origin/main`; this log and `docs/TWO_FLIES_PLAN.md` created.
- pytest, the four baseline experiment runs, the female build, the three games and the `Ctrl+C` check done (measurements above).
- Written: `tools/bench_two_flies.py` (from the plan), `tools/golden_hashes.py` (holds the shared CONFIGS), `tools/compare_experiments.py`,
  `tests/test_golden_single_fly.py` + `tests/golden_single_fly.json`; ARCHITECTURE.md's module tree lists them.
- Benchmark run twice (`../runs/p0-bench-first-run.json`, then `../runs/p0-bench.json` with the per-row processes); real-data
  golden hashes saved; SCIENCE.md 9.6 written; pushed; draft PR #16 opened.
