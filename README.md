# Entropy Robustness Index — Progressive-Collapse Robustness for Steel Frames (FEM, Python)

> A calibration-free, dimensionless **structural robustness** index for planar
> steel **frames and trusses**, computed from the **Shannon entropy of the
> strain-energy distribution** under the **alternate-load-path** (notional
> member-removal) procedure. Built on a verified linear-elastic
> **finite element** (Euler-Bernoulli) solver, in pure Python.

[![tests](https://github.com/fcarvajalbrown/Entropy-Collapse-Simulator/actions/workflows/tests.yml/badge.svg)](https://github.com/fcarvajalbrown/Entropy-Collapse-Simulator/actions/workflows/tests.yml)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](pyproject.toml)
[![validation](https://img.shields.io/badge/solver%20validation-0.0000%25-brightgreen.svg)](validation/validation_report.md)

**What it does, in one line:** tell whether a frame can lose a member without
collapsing, with a single number `R_S` in `[0, 1]` that needs no threshold
calibration.

Three analysis modes from one CLI: a **progressive-collapse simulation** with
entropy-based detection, the **Entropy Robustness Index** by member removal, and
a **head-to-head comparison** of the displacement, demand-capacity-ratio (DCR),
energy, and entropy collapse criteria. The full mathematical basis is in
[`THEORY.md`](THEORY.md); the journal manuscript is in [`manuscript/`](manuscript/).

> Keywords: structural engineering, finite element method, progressive collapse,
> structural robustness, redundancy, alternate load path, Shannon entropy,
> truss solver, frame analysis, reproducible research.

---

## Results at a glance

| Structure | Robustness `R_S` | Reading |
|---|---|---|
| Two-span beam (non-redundant) | **0.00** | any member loss collapses it |
| 2-bay 3-story moment frame | **0.72** | redundant; no single loss is fatal |
| X-braced truss bridge | **0.79** | redundant; worst single loss localizes most |

- **Verified:** matches closed-form solutions and an independent solver to **0.0000%**.
- **Novel, with evidence:** `R_S` rises monotonically with the degree of static
  indeterminacy, and its member-criticality ranking is *not* a re-encoding of the
  usual compliance-based importance (the two even anti-correlate for multi-bay
  frames).
- **Reproducible:** every number in the paper is computed at build time; 10 test
  phases; pinned dependencies.

---

## Contribution

Shannon entropy of strain/strain-energy distributions has been used before for
metamaterials, fatigue and vibration-based damage detection. The contribution
here is narrower and practice-oriented:

1. **Entropy Robustness Index `R_S`** — a calibration-free, dimensionless
   redundancy measure in `[0, 1]` obtained by combining the normalized
   strain-energy entropy with the alternate-load-path method (GSA 2003;
   UFC 4-023-03). `R_S -> 1` is robust (a single member loss barely changes how
   evenly energy is shared); `R_S -> 0` is fragile.
2. **A four-criteria comparison** — entropy versus the displacement, DCR and
   energy criteria on the same incremental analysis, extending the comparative
   study of Feng et al. (2024), who did not include an information-theoretic
   criterion.
3. **Verified engine** — exact agreement with closed-form solutions and an
   independent dual solver (Section *Validation*).

The key practical point: the displacement, DCR and energy criteria each need a
structure-specific threshold; `R_S` and the entropy criterion do not.

---

## The index in one equation

For the normalized strain-energy distribution `p_i = U_i / sum(U)` over the `N`
active members, the normalized entropy is

```
H = -sum_i p_i ln(p_i) / ln(N)      in [0, 1]
```

Notionally remove each primary member `k`, re-analyse, and record the survivors'
normalized entropy `H_k`. Then

```
R_S = mean_k H_k          (Entropy Robustness Index)
```

with `H_k = 0` when removing `k` turns the frame into a mechanism.

---

## Scope

- **Planar (2D)** frames in the global X-Y plane.
- **First-order, linear-elastic, quasi-static** analysis.
- Euler-Bernoulli frame element with axial and in-plane bending stiffness.

Out of scope (see [`THEORY.md`](THEORY.md), *Limitations*): plasticity,
second-order/P-Delta and catenary action, dynamics, and connection failure.
Linear-static alternate-load-path analysis is an accepted (conservative) code
method; the tool does not claim to reproduce the large-deformation reserve of
physical column-loss tests, and it does not claim predictive lead time over the
standard criteria.

---

## Installation

```bash
pip install -r requirements.txt   # numpy, matplotlib, Pillow
```

---

## Usage

Three analysis modes via `--mode`:

```bash
# List frames
python main.py --list

# 1) Progressive-collapse simulation with entropy-based detection
python main.py --mode simulate --scenario pratt_bridge --save

# 2) Entropy Robustness Index (alternate load path)
python main.py --mode robustness --scenario building_2d

# 3) Four-criteria comparison
python main.py --mode criteria --scenario pratt_bridge --steps 40 --load-step 0.15
```

Selected arguments:

| Flag | Default | Description |
|---|---|---|
| `--mode` | `simulate` | `simulate`, `robustness`, or `criteria` |
| `--scenario` | `2d_simple` | `2d_simple`, `building_2d`, `pratt_bridge` |
| `--method` | `zscore` | Detection for simulate mode: `zscore` or `threshold` |
| `--steps` | `100` | Max steps (simulate / criteria) |
| `--load-step` | `0.2` | Load-factor increment per step |
| `--save` | off | Save figures to `output_figures/` |

---

## Scenarios

| Scenario | Description | R_S |
|---|---|---|
| `2d_simple` | Two-span beam (frame elements). Non-redundant: any member loss collapses it. | 0.00 |
| `building_2d` | Planar 2-bay, 3-story steel moment frame. Redundant; column-removal demo. | 0.72 |
| `building_large` | Larger 3-bay, 6-story moment frame (28 nodes, 42 members). Scalability case. | 0.78 |
| `pratt_bridge` | 6-panel 30 m pin-jointed truss, X-braced to be statically indeterminate (redundant). | 0.79 |

---

## Project structure

```
core/            # shared dataclasses (Material, Node, Member, FrameData, ...)
structure/
  stiffness.py   # planar Euler-Bernoulli assembly, transformation, BCs
  frames/        # 2d_simple, building_2d, pratt_bridge
solver/
  equilibrium.py # Ku=F solve, per-member strain energy (solve / solve_full)
  failure.py     # combined axial+bending stress failure criterion
entropy/
  metrics.py     # S, dS, normalized entropy, max entropy
  localization.py# causal z-score / threshold detection, Gini index
  robustness.py  # Entropy Robustness Index R_S (alternate load path)
analysis/
  criteria.py    # four-criteria head-to-head comparison
  importance.py  # dH_k vs compliance importance (Spearman correlation)
  parametric.py  # R_S vs static indeterminacy; threshold/step-size sweeps
simulation/
  runner.py      # progressive-collapse loop (member removal + re-analysis)
  scenarios.py   # scenario registry
visualization/   # frame view, entropy plots, animation
benchmark.py     # validation harness (analytical + independent dual solver)
THEORY.md        # mathematical basis
main.py          # CLI entry point
```

All modules communicate only through `core/models.py` dataclasses.

---

## Validation

```bash
python benchmark.py            # prints tables, writes validation/validation_report.md
python benchmark.py --figures  # also saves output_figures/validation_errors.png
```

Two independent tiers:

**Tier 1 - Analytical (exact).** The displacement-method element is exact for
nodal point loads, so closed-form cases match to floating point:

| Case | Displacement error | Strain-energy error |
|---|---|---|
| Simply-supported beam (`PL^3/48EI`) | 0.0000% | 0.0000% |
| Fixed-fixed beam (`PL^3/192EI`) | 0.0000% | 0.0000% |
| Cantilever (`PL^3/3EI`) | 0.0000% | 0.0000% |

**Tier 2 - Independent dual solver.** A from-scratch 3-DOF-per-node planar
frame solver (no shared code) reproduces the redundant frames:

| Frame | Displacement error | Strain-energy error |
|---|---|---|
| 2D Moment Frame (2-bay, 3-story) | 0.0000% | 0.0000% |
| 2D Moment Frame (3-bay, 6-story) | 0.0000% | 0.0000% |
| Redundant truss bridge | 0.0000% | 0.0000% |

A third tier validates the quantities `R_S` actually uses: the per-member
distribution `{p_i}`, post-removal alternate-load-path states, the failure
criterion, and the stability (mechanism) test, all against independent
calculations. The Ziemian & Ziemian steel benchmark frames (Data in Brief, 2021) are cited as
a recognized external reference; their verified results are second-order
(P-Delta), outside this first-order solver's modelling scope, and are
recommended for future cross-checks rather than reproduced here.

---

## Tests

```bash
python tests/run_all_tests.py
```

Ten phases: models, stiffness, solver, failure/re-analysis, entropy metrics,
simulation, visualization, robustness index, criteria comparison, and novelty
studies (importance correlation, redundancy sweep, step-size sensitivity).

---

## References

- C. E. Shannon (1948). *A mathematical theory of communication.* Bell System Technical Journal, 27, 379-423.
- GSA (2003). *Progressive Collapse Analysis and Design Guidelines for New Federal Office Buildings.*
- DoD (2016). *UFC 4-023-03: Design of Buildings to Resist Progressive Collapse.*
- D. Feng et al. (2024). *Physically-based collapse failure criteria in progressive collapse analyses of multi-story RC structures under column removal.* Engineering Structures. https://doi.org/10.1016/j.engstruct.2024.119412
- C. W. Ziemian, R. D. Ziemian (2021). *Steel benchmark frames for structural analysis and validation studies.* Data in Brief, 39, 107564. https://doi.org/10.1016/j.dib.2021.107564

See [`THEORY.md`](THEORY.md) for the complete derivation and limitations.

---

## License

GNU General Public License v3.0 or later (GPL-3.0-or-later). See [`LICENSE`](LICENSE).

## Citation

If you use this software, please cite the accompanying manuscript (in
[`manuscript/`](manuscript/)): Carvajal Brown, F. (2026). *An Entropy Robustness
Index for Planar Steel Frames.* Revista de la Construcción (under review).
