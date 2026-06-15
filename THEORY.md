# Theoretical Basis of the Entropy Robustness Index

This document is the mathematical companion to the software. It states the
governing equations, the information-theoretic robustness measure that is the
project's contribution, the four collapse criteria used for comparison, and an
explicit statement of scope and limitations. Symbols used in the code are noted
in `monospace`.

---

## 1. Scope and modelling assumptions

The solver performs **first-order, linear-elastic, quasi-static** analysis of
**planar (2D) frames**. Each member is an Euler-Bernoulli frame element with
axial and in-plane bending stiffness; deformation occurs in the global X-Y
plane (bending about the global Z axis). Out-of-plane translation and rotation
and torsion are not modelled and must be restrained at every node
(`fixed_dofs` containing 2, 3, 4).

The following are intentionally **out of scope** and are discussed as
limitations in Section 8: material nonlinearity (plasticity), geometric
nonlinearity (second-order / P-Delta and catenary action), dynamic effects
(inertia, dynamic amplification), and connection/joint failure.

---

## 2. Linear-elastic frame analysis

### 2.1 Element

For a member of length `L`, axial rigidity `EA` and flexural rigidity `EI`, the
12x12 local stiffness matrix couples the axial degrees of freedom and the
in-plane bending degrees of freedom (the remaining out-of-plane entries are
zero, consistent with the planar scope):

```
k_axial      = EA/L
k_bending: 12EI/L^3, 6EI/L^2, 4EI/L, 2EI/L  (standard Euler-Bernoulli terms)
```

The element is mapped to global coordinates by a transformation `T` whose local
z-axis is fixed to the global Z axis for every member. This is the defining
choice of a planar frame element: it guarantees that the single bending plane
is the global X-Y plane for columns, beams and diagonals alike. (Selecting the
bending plane from a per-member reference vector — a common 3D heuristic —
silently rotates a column's bending plane out-of-plane and produces a singular
in-plane system; the present formulation avoids this.)

### 2.2 Assembly and solution

The global stiffness matrix is assembled from the non-failed members:

```
K = sum_e  T_e^T k_e T_e
```

Boundary conditions are enforced and the equilibrium system is solved:

```
K u = F
```

where `F` is the nodal load vector (scaled by a load factor `lambda` during
incremental analysis) and `u` the nodal displacement vector.

### 2.3 Member strain energy

The elastic strain energy stored in member `i` is computed from its local
displacements and internal forces:

```
U_i = 1/2 * u_local^T k_local u_local = 1/2 * u_local^T f_local
```

This captures axial and bending contributions together. The total strain
energy is `U = sum_i U_i`. For a fixed topology, `U` and each `U_i` scale with
`lambda^2`; the *fractions* defined next are therefore scale-invariant.

### 2.4 Failure criterion

A member is removed when its maximum combined stress reaches the yield stress:

```
sigma_max = |N|/A + |M_max| c / I  >=  sigma_y
```

with `N` the axial force, `M_max` the larger end moment, `c` the explicit
extreme-fibre distance (a true section property, `Material.c`), `A` the area
and `I` the second moment of area. The demand-capacity ratio is
`DCR = sigma_max / sigma_y`.

---

## 3. Strain-energy entropy

Define the normalized strain-energy distribution over the `N` active
(non-failed) members:

```
p_i = U_i / sum_j U_j ,   sum_i p_i = 1 ,   p_i >= 0
```

The **Shannon entropy** of this distribution (in nats) is

```
S = - sum_i p_i ln(p_i)        (entropy/metrics.py)
```

with the convention `0 ln 0 = 0`. `S` measures how evenly strain energy is
shared among members:

- maximum `S = ln(N)` when energy is perfectly uniform (`p_i = 1/N`);
- minimum `S = 0` when all energy is in a single member.

The **normalized entropy**

```
H = S / ln(N)  in  [0, 1]        (entropy/metrics.normalized_entropy)
```

removes the dependence on member count, allowing structures (and damaged
states with different `N`) to be compared on a common [0, 1] scale.

A complementary concentration measure is the **Gini coefficient** of `{p_i}`
(`entropy/localization.localization_index`), with 0 for a uniform distribution
and approaching 1 for a fully localized one.

Because `{p_i}` is scale-invariant, `S` and `H` do **not** change as the load
factor increases on a fixed topology. They change only when the load path
changes — i.e. when a member is removed. This is the property exploited below.

---

## 4. Alternate load path by re-analysis

Progressive-collapse guidance (e.g. GSA 2003; UFC 4-023-03) assesses robustness
by the **alternate load path (ALP)** method: a primary member is notionally
removed and the structure is re-analysed to check whether the remaining members
can carry the load. In a linear-elastic model the redistribution after removal
is obtained *exactly* by excluding the member from the assembly and re-solving
`K u = F`. No separate, tuned redistribution law is needed or used; the
re-analysis **is** the redistribution.

### 4.1 Kinematic stability

Removing a member may turn the frame into a mechanism. Stability is tested on
the free-DOF partition `K_ff` of the stiffness matrix: the state is stable iff
`K_ff` is positive definite (smallest eigenvalue above a relative tolerance).
An unstable removal means the load cannot be redistributed — the member is
maximally critical.

---

## 5. The Entropy Robustness Index

Let `H_0` be the normalized entropy of the intact frame under the design load.
For each member `k` in a set `R` of primary members, notionally remove `k`,
re-analyse, and record the normalized entropy `H_k` of the survivors (set
`H_k = 0` if the removal causes a mechanism). Define the per-member **entropy
drop**

```
dH_k = H_0 - H_k
```

A large `dH_k` means losing member `k` forces energy to concentrate into few
members (or collapses the structure) — `k` is critical. The **Entropy
Robustness Index** is the mean post-removal normalized entropy:

```
R_S = (1/|R|) sum_{k in R} H_k   in  [0, 1]     (entropy/robustness.analyze)
```

Interpretation:

- `R_S -> 1`: losing any single member barely changes how evenly energy is
  shared — high redundancy, robust;
- `R_S -> 0`: a single loss funnels energy into few members or collapses the
  frame — low redundancy, fragile.

Auxiliary outputs:

- **Worst case** `min_k H_k` — the most damaging single loss;
- **Critical member** `argmax_k dH_k`;
- **Criticality ranking** — members ordered by `dH_k`, an entropy-based member
  importance measure.

### 5.1 Properties

1. **Dimensionless and bounded.** `R_S in [0, 1]`, independent of units and of
   absolute load magnitude (a consequence of the scale-invariance of `{p_i}`).
2. **Calibration-free.** `R_S` requires no structure-specific threshold (no
   drift limit, acceptance DCR, or energy multiple). This is its main practical
   advantage over the criteria in Section 6.
3. **Topology-aware.** `R_S` depends on the load paths, not on the load level,
   so a single assessment characterizes the structure rather than a particular
   loading.

### 5.2 Sequential (progressive) trajectory

Removing members one at a time (in a prescribed order, or greedily by largest
`dH_k`) and recording `H` after each removal yields an **entropy collapse
trajectory** (`entropy/robustness.sequential_trajectory`). It traces how
quickly redundancy is exhausted along a chosen failure path and complements the
single-removal index.

---

### 5.3 Relationship to static indeterminacy

`R_S` is bounded below by the structure's redundancy in a precise sense.

**Proposition (determinacy bound).** Consider a stable pin-jointed truss with
`m` bars, `r` support reactions, `n` nodes, and degree of static
indeterminacy `d = m + r - 2n`. If `d = 0` (statically determinate), then the
removal of any single bar produces a mechanism; consequently `H_k = 0` for
every `k` and `R_S = 0`. Equivalently, `R_S > 0` implies `d >= 1`.

*Proof.* A stable, statically determinate truss has `m + r = 2n` with a
non-singular equilibrium (and stiffness) matrix. Removing one bar leaves
`m - 1` bars, so `(m - 1) + r = 2n - 1 < 2n`: the number of independent
internal-force and reaction unknowns is now smaller than the number of nodal
equilibrium equations, so no equilibrium state carries a general load and the
reduced free-DOF stiffness matrix is singular. Hence `is_stable` returns False,
the convention sets `H_k = 0`, and since this holds for every bar,
`R_S = mean_k H_k = 0`. The contrapositive gives `R_S > 0 => d >= 1`. ∎

This is the analytic counterpart of the parametric result in
`analysis/parametric.py`: starting from the determinate single-diagonal Pratt
truss (`d = 0`, `R_S = 0`) and adding counter-diagonals one at a time
(`d = 1, 2, ..., 6`), the index rises monotonically toward the fully X-braced
value while the fraction of single-loss mechanisms falls to zero. `R_S`
therefore increases with redundancy rather than merely re-encoding stiffness or
strength, which is the behaviour a robustness measure should have. The bound
also clarifies the role of the `H_k = 0` mechanism convention (Section 5): it is
exactly what makes `R_S` sensitive to the *survivability* of removals, not only
to the shape of the surviving distribution (reported separately as
`R_S` over stable removals only).

## 6. Four collapse criteria for comparison

For a fair comparison the four criteria are evaluated on the *same* incremental
analysis (`analysis/criteria.py`). At each load step the frame is solved, over-
stressed members are removed, and four indicators are recorded. Each criterion
fires at the first step that satisfies its condition.

| Criterion | Indicator | Fires when | Threshold needed |
|---|---|---|---|
| Displacement | peak nodal translation | exceeds a drift limit | yes (drift limit) |
| DCR | max demand-capacity ratio | reaches yielding (DCR >= 1) | yes (acceptance DCR) |
| Energy | compliance `U / lambda^2` | exceeds a multiple of the intact value | yes (energy multiple) |
| Entropy | drop in `H` | statistical outlier (causal z-score) | no |

The **energy** indicator is the compliance `U/lambda^2` rather than `U` itself,
because `U` scales with `lambda^2` for a fixed topology and would otherwise fire
trivially under load increase; compliance is constant for the intact elastic
frame and jumps only when members soften or are removed.

The **entropy** criterion uses a causal z-score test: a step fires when its
entropy drop `dS` lies more than `z*` standard deviations below the mean of the
strictly preceding steps. Because the entropy is flat (scale-invariant) until
the first failure, a clearly negative drop from a flat baseline is treated as
significant. The test uses only past information and needs no per-structure
calibration.

This comparison extends the displacement/resistance/energy comparison of Feng
et al. (2024) by adding the information-theoretic criterion they did not
consider.

---

## 7. Progressive collapse simulation

The full simulation (`simulation/runner.py`) repeats, under an optionally
increasing load factor:

1. solve `K u = F` for the current (possibly damaged) topology;
2. compute `S`, `dS` and `H`;
3. test for collapse via the causal entropy z-score detector;
4. remove any members that reach the failure criterion;
5. repeat until collapse is detected, all members fail, or the step budget is
   exhausted.

Removed members are simply excluded from the next assembly, so step 1 of the
following iteration performs the exact alternate-load-path redistribution.

---

## 8. Limitations and honest caveats

- **Linear elastic, first order.** No plasticity, no second-order (P-Delta) or
  catenary action. Linear-static ALP is an accepted (and generally
  conservative) method in the codes, but it does not capture the large-
  deformation reserve strength seen in physical column-loss tests.
- **Planar.** Out-of-plane and torsional behaviour are not modelled.
- **Quasi-static.** No inertia or dynamic amplification.
- **Member-level failure.** Connection and joint behaviour are not represented.
- **No guaranteed lead time.** The entropy drop is a *coincident* indicator of
  load-path localization, not a proven early-warning signal. Recent work on
  feedback-amplified systems shows entropy collapse can be a first-order
  transition with no statistical precursor; we therefore claim a calibration-
  free, dimensionless robustness/localization measure, not predictive lead time
  over the standard criteria.

These constraints are appropriate for a transparent research tool whose purpose
is to formulate and demonstrate the entropy robustness measure. Extension to
nonlinear, dynamic and 3D analysis is the natural next step.

---

## 9. Verification and validation

- **Analytical (exact).** Simply-supported, fixed-fixed and cantilever beams
  reproduce the closed-form deflection and strain energy to floating-point
  accuracy (the displacement-method element is exact for nodal point loads).
- **Independent dual solver.** A from-scratch 3-DOF-per-node planar frame
  solver (`benchmark.py: independent_solve`), sharing no code with the
  production assembler, reproduces joint displacements and total strain energy
  of the redundant benchmark frames.

See `validation/validation_report.md` (generated by `python benchmark.py`).

---

## Relation to prior work

The principle that an even internal distribution signals robustness predates
this index and has been stated in information-theoretic terms before: the
redundancy matrix and its homogeneous-redundancy criterion (von Scheven et al.,
2021), Nafday's (2011) information-theoretic structural integrity, the
event-oriented definition of redundancy as the entropy of operational modes
(Ziha, 2000), and the entropy of load-flow distributions in
power-grid cascade analysis (Koc et al., 2013). Strain-energy member-importance
coefficients are also established (Lin, 2019). `R_S` is a specific recombination
of these ideas: the entropy of the member-level elastic strain-energy
distribution, under code-style single-member removal, on planar frames. Its
distinguishing, demonstrated properties are the determinacy bound (Section 5.3),
the monotonic tracking of static indeterminacy, and the non-reducibility to
compliance-based importance (the two can anti-correlate), shown in
`analysis/parametric.py` and `analysis/importance.py`.

## References

- C. E. Shannon (1948). *A mathematical theory of communication.* Bell System
  Technical Journal, 27, 379-423.
- A. M. Nafday (2011). *Consequence-based structural design approach for black
  swan events.* Structural Safety, 33(1), 108-114.
- K. Ziha (2000). *Redundancy and robustness of systems of events.*
  Probabilistic Engineering Mechanics, 15(4), 347-357.
- Y. Koc, M. Warnier, R. E. Kooij, F. M. Brazier (2013). *An entropy-based
  metric to quantify the robustness of power grids against cascading failures.*
  Safety Science, 59, 126-134.
- M. von Scheven, E. Ramm, M. Bischoff (2021). *Quantification of the redundancy
  distribution in truss and beam structures.* Int. J. Solids Struct., 213, 41-49.
  https://doi.org/10.1016/j.ijsolstr.2020.11.002
- K. Lin et al. (2019). *Importance assessment of structural members based on
  elastic-plastic strain energy.* Advances in Materials Science and Engineering,
  2019, 8019675.
- D. M. Frangopol, J. P. Curley (1987). *Effects of damage and redundancy on
  structural reliability.* Journal of Structural Engineering, 113(7), 1533-1549.
- M. Ghosn, F. Moses, D. M. Frangopol (2010). *Redundancy and robustness of
  highway bridge superstructures and substructures.* Structure and
  Infrastructure Engineering, 6(1-2), 257-278.
- General Services Administration (2003). *Progressive Collapse Analysis and
  Design Guidelines for New Federal Office Buildings and Major Modernization
  Projects.*
- Department of Defense (2016). *Unified Facilities Criteria UFC 4-023-03:
  Design of Buildings to Resist Progressive Collapse.*
- D. Feng et al. (2024). *Physically-based collapse failure criteria in
  progressive collapse analyses of random-parameter multi-story RC structures
  subjected to column removal scenarios.* Engineering Structures.
  https://doi.org/10.1016/j.engstruct.2024.119412
- C. W. Ziemian, R. D. Ziemian (2021). *Steel benchmark frames for structural
  analysis and validation studies: Finite element models and numerical
  simulation data.* Data in Brief, 39, 107564.
  https://doi.org/10.1016/j.dib.2021.107564
- A. Moreno-Gomez et al. (2018). *EMD-Shannon entropy-based methodology to
  detect incipient damages in a truss structure.* Applied Sciences, 8(11), 2068.
- T.-K. Lin, A. G. Lainez (2018). *Entropy-based structural health monitoring
  system for damage detection in multi-bay three-dimensional structures.*
  Entropy, 20(1), 49.
