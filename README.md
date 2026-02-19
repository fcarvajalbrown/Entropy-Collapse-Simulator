# Entropy-Based Progressive Collapse Simulator

A structural engineering research tool that uses **Shannon entropy of strain energy distribution** as a collapse predictor for frame structures under progressive member failure. Built on a full 3D Euler-Bernoulli FEM solver with modular architecture designed for extensibility and scientific reproducibility.

---

## Motivation

Current practice in progressive collapse analysis relies primarily on **displacement-based criteria** — structures are flagged as collapsed when nodal displacements exceed empirical thresholds (typically θ = 0.15–0.20 rad chord rotation per GSA guidelines). A comprehensive 2024 review by Feng et al. in the *Journal of Building Engineering* notes that no universally accepted collapse criterion exists, with researchers using displacement, resistance, and energy-based approaches inconsistently across the literature.

This tool proposes **structural entropy** as a scalar, dimensionless collapse indicator:

```
S = -Σ pᵢ ln(pᵢ)     where pᵢ = Uᵢ / Σ U
```

Where `Uᵢ` is the strain energy in member `i`. When the structure is healthy, strain energy is distributed across members (high entropy). As failure progresses and energy localizes into fewer members, entropy drops sharply. A large negative spike in `dS/dt` signals imminent collapse — potentially earlier than displacement-based detection.

This approach is distinct from vibration-based Shannon entropy methods used in structural health monitoring (Moreno-Gomez et al., 2018; Lin & Laínez, 2018), which apply entropy to acceleration time-series rather than static strain energy fields. The application of Shannon entropy to quasi-static strain energy redistribution during progressive collapse is the novel contribution of this tool.

---

## Key Features

- Full 3D Euler-Bernoulli FEM solver with correct axial + bending stiffness assembly
- Combined axial and bending stress failure criterion: `σ_max = |N|/A + |M_max| · c / I`
- Per-step entropy computation: S, dS/dt, normalized entropy, Gini index
- Two collapse detection methods: z-score (adaptive) and threshold (simple)
- 3D visualization with strain energy heatmap per member
- Entropy evolution plots with collapse marker
- Material dataclass (`Material`) for scientifically rigorous per-member property definition
- Standalone `.exe` build via PyInstaller

---

## Project Structure

```
entropy_collapse_simulator/
├── core/
│   └── models.py           # Shared dataclasses: Material, Node, Member, FrameData, etc.
├── structure/
│   ├── stiffness.py         # Global K assembly, transformation matrix, BCs
│   └── frames/
│       ├── frame_2d_simple.py       # 3-node 2-member simply-supported beam
│       ├── frame_3d_redundant.py    # 5-node 8-member space frame with apex load
│       └── frame_pratt_bridge.py    # 14-node 25-member 6-panel Pratt truss (30m span)
├── solver/
│   ├── equilibrium.py       # Ku=F solver, full 12-DOF strain energy computation
│   ├── failure.py           # Combined stress failure criterion, member marking
│   └── redistribution.py    # ODE-based energy transfer after failure
├── entropy/
│   ├── metrics.py           # S, dS/dt, normalized entropy
│   └── localization.py      # Collapse detection (zscore/threshold), Gini index
├── simulation/
│   ├── runner.py            # Main simulation loop
│   └── scenarios.py        # Scenario registry
├── visualization/
│   ├── graph_view.py        # 3D frame viewer with energy heatmap
│   ├── entropy_plot.py      # S, dS/dt, Gini index plots
│   └── animation.py         # Animated entropy evolution (GIF/MP4)
├── tests/                   # 7-phase test suite (29 tests, all passing)
├── main.py                  # CLI entry point
└── requirements.txt
```

All modules communicate only through `core/models.py` dataclasses — no cross-module imports.

---

## Installation

```bash
pip install -r requirements.txt
```

**Requirements:** `numpy`, `scipy`, `matplotlib`, `networkx`, `pyinstaller`

---

## Usage

```bash
# List available scenarios
python main.py --list

# Run a scenario (displays plots interactively)
python main.py --scenario 2d_simple
python main.py --scenario 3d_redundant --method threshold --steps 200
python main.py --scenario pratt_bridge --steps 100

# Save figures to output_figures/
python main.py --scenario pratt_bridge --save

# Produce an animation of the entropy evolution
python main.py --scenario pratt_bridge --animate --save

# Animation with custom settings
python main.py --scenario pratt_bridge --animate --animate-fmt mp4 --fps 15 --save

# Control incremental loading (higher load-step = faster failure onset)
python main.py --scenario pratt_bridge --load-step 0.2 --animate --save
```

**Arguments:**

| Flag | Default | Description |
|---|---|---|
| `--scenario` | `2d_simple` | Frame to simulate |
| `--method` | `zscore` | Collapse detection: `zscore` or `threshold` |
| `--steps` | `100` | Maximum simulation steps |
| `--load-step` | `0.2` | Load factor increment per step. Set to `0.0` for static loading at design load. Increase to drive faster progressive failure. |
| `--save` | off | Save all figures to `output_figures/` |
| `--animate` | off | Produce an animation of the entropy evolution (always saved to file) |
| `--animate-fmt` | `gif` | Animation format: `gif` (requires Pillow) or `mp4` (requires ffmpeg) |
| `--fps` | `10` | Animation frames per second |

### Animation Output

The animation shows three entropy metrics evolving step by step with a sweeping marker:

1. **S / S_max** — normalized structural entropy. Drops as energy localizes.
2. **dS / dt** — rate of change. A large negative spike signals imminent collapse.
3. **Gini index** — energy concentration measure. Rises toward 1.0 as collapse approaches.

Member failure events are marked with grey dotted lines. The detected collapse step is marked with a red dashed line. Output file is saved to `output_figures/collapse_<scenario>.gif` (or `.mp4`) when `--save` is set, otherwise to the current directory.

**GIF dependency:**
```bash
pip install Pillow
```

**MP4 dependency:** Install [ffmpeg](https://ffmpeg.org/) and ensure it is on your system PATH.

---

## Frames

### 2D Simple Truss
3-node, 2-member simply-supported beam. 50 kN midspan load. Used for baseline validation — verifies that bending-dominated response is correctly captured (a common FEM pitfall for horizontal members under vertical load).

### 3D Redundant Space Frame
5-node, 8-member pyramid frame with 4 pinned base corners and a free apex. 200 kN downward apex load. Tests energy redistribution across multiple load paths — the redundancy keeps entropy high until multiple members fail.

### Pratt Truss Bridge
14 nodes, 25 members, 6-panel 30m span with differentiated materials per member type:

| Member type | Material | Rationale |
|---|---|---|
| Bottom chord | S355, A=0.0155 m² | Tension dominant, W360×122 equivalent |
| Top chord | S355, A=0.0123 m² | Compression dominant, W310×97 equivalent |
| Verticals | S275, A=0.0066 m² | Secondary members, W200×52 equivalent |
| Diagonals | S355, A=0.0114 m² | Primary load path, W250×89 equivalent |

600 kN total distributed traffic load (100 kN per interior bottom chord node, 50 kN at supports). Under standard S355 steel, no collapse occurs — this is physically correct. Reduce material grade to observe progressive failure sequence and entropy localization.

---

## Adding New Frames

1. Create `structure/frames/frame_name.py`
2. Define materials using `dataclasses.replace(STEEL_S275, ...)` or a custom `Material`
3. Implement `build() -> FrameData` with nodes, members, loads
4. For 2D frames in the XY plane: add `fixed_dofs=[2, 3, 4]` to all nodes to constrain out-of-plane DOFs
5. Register in `simulation/scenarios.py` and `main.py`

**Important:** Loads applied at supported nodes are automatically zeroed during solve (boundary condition enforcement). This is correct FEM behavior — do not apply loads at pinned/roller support nodes unless you intend to load them before constraints are applied.

---

## Collapse Detection Methods

**Z-score (recommended for research):** Flags collapse when `dS` deviates beyond N standard deviations from the rolling mean. Adaptive — does not require manual threshold calibration.

**Threshold:** Flags collapse when `dS < -threshold`. Simple and fast, requires calibration per frame.

---

## Scientific Context

### Relation to Existing Literature

**Displacement-based criteria** (GSA 2003, UFC 4-023-03) remain the standard in practice and are well-studied for RC frames (Feng et al., 2024; Parisi et al., 2020). They are empirically validated but require structure-specific calibration of drift limits.

**Energy-based methods** have been applied to dynamic progressive collapse analysis via the Energy Balance Method (EBM), where work done by gravity loads is compared to strain energy capacity (Feng et al., 2024). This tool extends the energy-based philosophy into the information-theoretic domain — rather than comparing total energy to a capacity, it measures the *distribution* of energy across members.

**Entropy in structural engineering** has been applied primarily to vibration signal processing for damage detection (Moreno-Gomez et al., *Applied Sciences*, 2018; Lin & Laínez, *Entropy*, 2018; Amezquita-Sanchez et al., 2021). These methods apply entropy to time-domain acceleration signals, not to static strain energy fields. The cross-sample entropy SHM system by Lin & Laínez (2018) demonstrated damage localization in 3D multi-bay frames using MSCE of floor response signals, establishing entropy as a viable structural indicator.

**This tool's novel position:** Applying Shannon entropy directly to the quasi-static strain energy distribution `{pᵢ}` during progressive member removal. The Gini index of `{pᵢ}` supplements entropy as a concentration measure, giving both a mean-field view (S) and an inequality measure (Gini) of the structural state.

### Limitations

- Linear elastic only — no plasticity, catenary action, or geometric nonlinearity
- Quasi-static — no dynamic amplification (no DAF)
- 6-DOF Euler-Bernoulli elements — no shear deformation (Timoshenko), no torsion
- Failure criterion is member-level stress — no connection or joint failure
- Energy redistribution via ODE coupling is phenomenological, not derived from equilibrium

These are appropriate simplifications for a research prototype demonstrating the entropy indicator concept. Extension to nonlinear dynamic analysis is the natural next step.

---

## Test Suite

7 phases, 29 tests:

```bash
python tests/run_all_tests.py
```

| Phase | Coverage |
|---|---|
| 1 — Models | Dataclass instantiation, Material properties, frame build() |
| 2 — Stiffness | K shape, symmetry, boundary condition enforcement |
| 3 — Solver | Displacement correctness, strain energy magnitude, 3D frame |
| 4 — Failure | Combined stress criterion, member flag, energy conservation |
| 5 — Entropy | S formula, dS sign, normalized entropy, Gini index |
| 6 — Simulation | End-to-end runs, collapse detection, failure sequence order |
| 7 — Visualization | Plot functions save without error, collapse overlay renders |

---

## Building a Standalone Executable

```bash
pip install pyinstaller
pyinstaller --onefile main.py
```

Output: `dist/main.exe` (Windows) or `dist/main` (Linux/macOS). No Python installation required on target machine.

---

## References

- Feng, D. et al. (2024). *Physically-based collapse failure criteria in progressive collapse analyses of random-parameter multi-story RC structures.* Journal of Building Engineering. https://doi.org/10.1016/j.jobe.2024.019412
- Moreno-Gomez, A. et al. (2018). *EMD-Shannon entropy-based methodology to detect incipient damages in a truss structure.* Applied Sciences, 8(11), 2068.
- Lin, T.-K. & Laínez, A.G. (2018). *Entropy-based structural health monitoring system for damage detection in multi-bay three-dimensional structures.* Entropy, 20(1), 49.
- Amezquita-Sanchez, J.P. et al. (2021). *Entropy algorithms for detecting incipient damage in high-rise buildings subjected to dynamic vibrations.* Journal of Vibration and Control.
- Shannon, C.E. (1948). *A mathematical theory of communication.* Bell System Technical Journal, 27, 379–423.
- GSA (2003). *Progressive Collapse Analysis and Design Guidelines for New Federal Office Buildings.*