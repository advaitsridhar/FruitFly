# Two flies: progress log

Plan: docs/TWO_FLIES_PLAN.md. Newest session notes first.

## Status
| phase | branch | PR | state | last update |
|---|---|---|---|---|
| 0 | claude/two-flies-p0-baseline | #? | in progress | 2026-09-27 |

## This machine
- OS: WSL2 (Ubuntu 24.04.5 LTS) on Windows; the repository lives under the Linux home folder, not `/mnt/c`.
- CPU: Intel Core i9-14900HX, 16 cores / 32 threads visible to WSL2.
- RAM: 15 GB visible to WSL2 (the WSL2 default cap), 4 GB swap. Free disk: about 950 GB.
- GPU: NVIDIA GeForce RTX 4070 Laptop GPU, 8,188 MiB, compute capability 8.9 (Ada). Windows driver 617.14
  (`nvidia-smi` reports NVIDIA-SMI 615.78.02, KMD 617.14); CUDA 13.4 shown by `nvidia-smi`. `/usr/lib/wsl/lib/libcuda.so`
  present. No `/dev/dri` (so no Mesa `d3d12` in WSL2: offline video will need `osmesa`, Phase 4).
  So: R580+ class driver, CUDA 13 wheels for Phase 2 (plan 4.3).
- Python 3.12.3 (the system `python3`; venv at `.venv/`, pip 26.2.1). git 2.43.0, gh 2.101.0 (logged in; `gh auth setup-git` done).
- All apt packages of plan 4.4 were already installed (git, curl, build-essential, python3-venv, python3-dev, libegl1, libgl1,
  libosmesa6, ffmpeg, gh): no `sudo` step was needed.
- Package versions (filled in after the install): see the session notes below.

## Fresh-clone findings
- `origin/main` at the start: `739e791 v2.8.1: fixes for everything a fresh-clone test of v2.8.0 found (#15)`. No neuPrint
  harvest bot commit on top. History below it as the plan describes (PR #12's two workflow files, PR #11 = v2.8.0).
- `pyproject.toml` version 2.8.1 and `virtual_fly/__init__.py` `__version__` 2.8.1: as expected.
- `stop_loop` in server.py, `def _guard` in connectome.py, `OverflowError` in game.py: 1, 1, 1 (the final v2.8.1).
- `GF_BURST = 5`, `STILL_TICKS = 8`, `flywire.BUILD = 6`; explicit `packages` list and `pythonpath` in pyproject.toml;
  `dev = ["pytest>=7", "numba>=0.59"]` (no Playwright); `FLY_DATA_DIR` unset.
- Workflows: ci.yml, claude-code-review.yml, claude.yml, neuprint-harvest.yml.
- None of the files the plan marks **new** existed on `main`; no `claude/two-flies-*` branch existed.
- One other remote branch, `claude/elegant-ramanujan-glf7g9`, is the merged source branch of PR #15 (its content is
  identical to `main`); `add-claude-github-actions-…` is PR #12's. Neither is an earlier attempt at this plan.
- `brain.py` rejects backends outside ("auto", "numpy", "numba"); `--backend` choices in cli.py and play.py match.
- The venv on Python 3.12 has no setuptools by default; pip's build isolation handles the editable installs.

## Decisions
| # (section 10) | question | answer | who | date |
|---|---|---|---|---|
| 1 | operating system route | WSL2 (Ubuntu 24.04), Claude Code inside Ubuntu: the machine already matched the default | default, taken in an autonomous session; the owner may override | 2026-09-27 |
| 2 | Python version | the system python3, 3.12.3 (in the 3.10-3.12 range) | default | 2026-09-27 |
| 3 | continue before a PR is merged | wait for the merge before the next phase | default | 2026-09-27 |

## Measurements
(none yet)

## Open issues
(none yet)

## Next step
Phase 0, step 7 (plan 4.7): run pytest, the four baseline experiment runs, build the female, try the three games.

## Session notes
### 2026-09-27
- Read the plan; machine facts gathered (4.1, 4.3); no missing system packages (4.4).
- Cloned at `739e791`; venv created; installs of the dev, female and physics extras and flygym 1.2.1 (`--no-deps`) started.
- Fresh-clone checks (4.6, first block) all as the plan expects; findings above.
- Branch `claude/two-flies-p0-baseline` created from `origin/main`; this log and `docs/TWO_FLIES_PLAN.md` created.
