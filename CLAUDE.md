# CLAUDE.md

Guidance for Claude Code (and humans) working in this repository. If anything
here conflicts with `README.md` or `THEORY.md`, those two win.

## What this project is

A planar (2D), first-order, linear-elastic Euler-Bernoulli frame analysis tool.
Its research contribution is the **Entropy Robustness Index `R_S`**: a bounded,
calibration-free redundancy measure computed from the Shannon entropy of the
strain-energy distribution, evaluated under the alternate-load-path (notional
member-removal) procedure. The tool also runs a progressive-collapse simulation
and a four-criteria comparison (entropy vs displacement, DCR, energy).

It is being prepared for submission to *Revista de la Construccion* (Journal of
Construction), PUC Chile. Deliverables: the software, `THEORY.md`, and the
manuscript under `manuscript/`.

## Commands

```bash
pip install -r requirements.txt          # numpy, matplotlib, Pillow
pip install python-docx                   # only to rebuild the Word manuscript

python main.py --list                                            # list scenarios
python main.py --mode simulate   --scenario pratt_bridge --save  # collapse sim + figures
python main.py --mode robustness --scenario building_2d          # Entropy Robustness Index
python main.py --mode criteria   --scenario pratt_bridge --steps 40 --load-step 0.15

python tests/run_all_tests.py            # full suite: 10 phases, must stay green
python tests/test_phase8_robustness.py   # run one phase directly
python benchmark.py                       # validation (analytical + independent solver)
python benchmark.py --figures             # also writes output_figures/validation_errors.png
python manuscript/generate_figures.py     # regenerate manuscript figures from live runs
python manuscript/build_docx.py           # rebuild manuscript/Carvajal_RDLC_manuscript.docx
```

There is no separate lint step. Tests are plain assert-based scripts (no pytest
required) run as subprocesses by `tests/run_all_tests.py`.

## Architecture

Data flows one way and every module communicates only through the dataclasses
in `core/models.py` (no cross-module internal imports):

```
frames -> stiffness -> solver -> entropy/analysis -> simulation -> visualization
```

- `core/models.py` — `Material`, `Node`, `Member`, `Load`, `FrameData`,
  `EnergyState`, `MemberState`, `EntropyRecord`, `SimulationResult`.
- `structure/stiffness.py` — global K assembly, planar transformation, BCs.
- `structure/frames/` — `frame_2d_simple`, `frame_building_2d`,
  `frame_pratt_bridge`; each exposes `build() -> FrameData`.
- `solver/equilibrium.py` — `solve` / `solve_full` (Ku=F + per-member strain
  energy); `solver/failure.py` — combined axial+bending stress criterion.
- `entropy/metrics.py` — S, dS, normalized entropy; `entropy/localization.py` —
  causal z-score / threshold detection, Gini; `entropy/robustness.py` — the
  `R_S` index (`analyze`, `is_stable`, `removal_entropy`, `sequential_trajectory`).
- `analysis/criteria.py` — four-criteria comparison (`run_with_metrics`, `compare`);
  `analysis/importance.py` — dH_k vs compliance importance (Spearman);
  `analysis/parametric.py` — R_S vs static indeterminacy, threshold/step sweeps.
- `simulation/runner.py` — progressive-collapse loop; `simulation/scenarios.py` —
  scenario registry.
- `visualization/` — frame view, entropy plots, animation.
- `benchmark.py` — independent dual solver + analytical verification.
- `manuscript/` — `manuscript.md` (reading copy), `build_docx.py` (authoritative
  RDLC Word output), `generate_figures.py`, `figures/`.

## Hard invariants (do not break)

- **Planar only.** All members lie in the z=0 plane and every node must fix the
  out-of-plane DOFs `[2, 3, 4]`. The element carries axial and in-plane bending
  (about global Z) only. `structure.stiffness._transformation_matrix` raises on
  any member with out-of-plane extent. Do not reintroduce a per-member
  reference-vector bending plane: it rotates a column's bending plane out of
  plane and leaves the in-plane system singular (a least-squares fallback then
  hides the error).
- **No phenomenological redistribution.** After a member fails it is excluded
  from the assembly and `Ku=F` is re-solved; that re-analysis is the exact
  alternate-load-path redistribution. The old `solver/redistribution.py` was
  removed on purpose. Do not add a tuned diffusion/coupling law.
- **Explicit section property.** Bending stress uses `Material.c` (the extreme-
  fibre distance), not `sqrt(I/A)` (the radius of gyration).
- **Scientific honesty for the paper.** "Entropy of strain energy" is not novel
  on its own; the contribution is the robustness formulation plus the verified
  implementation and comparison. Do not claim predictive early warning: the
  entropy drop is a coincident localization marker, and limitations stay
  explicit (planar, linear-elastic, quasi-static; no plasticity, second-order,
  catenary, dynamics, or connection failure).

## Conventions

- All code, comments, and docs are in English.
- Plain ASCII in source and docs (no emojis; avoid em-dashes, including in
  generated manuscript prose).
- Add new behaviour to the matching test phase under `tests/`; keep all 10
  phases green. Keep tests deterministic.
- Numbers in the README/manuscript are reproducible from `benchmark.py`,
  `main.py`, and `manuscript/generate_figures.py`; if behaviour changes,
  regenerate rather than hand-editing reported values.
- **Git commits:** plain messages, no AI co-author or attribution trailer ever
  (this overrides any default to add `Co-Authored-By`).

## Repo and license

- License: **GPL-3.0-or-later** (`LICENSE`).
- Remote: `github.com/fcarvajalbrown/Entropy-Collapse-Simulator` (branch `main`),
  pushed over HTTPS using `gh` as the git credential helper. CI is
  `.github/workflows/tests.yml` (runs the suite + `benchmark.py` on push/PR).

## Scenarios and reference values

| Scenario | What it is | R_S |
|---|---|---|
| `2d_simple` | two-span beam (non-redundant) | 0.00 |
| `building_2d` | 2-bay 3-story steel moment frame (redundant) | 0.72 |
| `building_large` | 3-bay 6-story moment frame (scalability) | 0.78 |
| `pratt_bridge` | 6-panel X-braced pin-jointed truss (redundant) | 0.79 |

Solver verification (`benchmark.py`) is exact: 0.0000% against three closed-form
beam cases and the independent dual solver, plus a third tier that checks the
per-member distribution, post-removal ALP states, the failure criterion, and
the stability test. Note: the bridge is now a genuine pin-jointed truss made
redundant with counter-diagonals (the old single-diagonal Pratt was relabeled
honestly; a determinate truss has R_S = 0 by the determinacy bound).
