# Entropy-Based Progressive Collapse Simulator — Agent Guide

This document contains project-specific context for AI coding agents. Read this first before modifying any code.

---

## Project Overview

This is a structural engineering research tool that uses **Shannon entropy of strain energy distribution** as a collapse predictor for frame structures under progressive member failure. It is built on a full 3D Euler-Bernoulli FEM solver with a modular architecture designed for extensibility and scientific reproducibility.

The central hypothesis: when a structure is healthy, strain energy is distributed across members (high entropy). As failure progresses and energy localizes into fewer members, entropy drops sharply. A large negative spike in `dS/dt` signals imminent collapse — potentially earlier than displacement-based detection.

**Key formula:**
```
S = -Σ pᵢ ln(pᵢ)     where pᵢ = Uᵢ / Σ U
```

Where `Uᵢ` is the strain energy in member `i`.

**Language:** All code, comments, and documentation are written in English.

---

## Technology Stack

- **Language:** Python 3.11+ (requires Python >=3.11 per `pyproject.toml`)
- **Core dependencies:** `numpy>=1.26`, `scipy>=1.12`, `matplotlib>=3.8`, `networkx>=3.2`, `Pillow>=10.0`
- **Build system:** `setuptools>=68` with `wheel`
- **Packaging:** Standard PEP 517/518 via `pyproject.toml`
- **Standalone executable:** PyInstaller (`pip install pyinstaller; pyinstaller --onefile main.py`)
- **Benchmark reporting:** Optional `reportlab` for PDF generation

---

## Project Structure

```
├── core/
│   └── models.py              # SHARED DATACLASSES — the only cross-module contract
├── structure/
│   ├── stiffness.py           # Global K assembly, transformation matrix, boundary conditions
│   └── frames/
│       ├── frame_2d_simple.py         # 3-node 2-member simply-supported beam
│       ├── frame_3d_redundant.py      # 5-node 8-member space frame with apex load
│       └── frame_pratt_bridge.py      # 14-node 25-member 6-panel Pratt truss (30m span)
├── solver/
│   ├── equilibrium.py         # Ku=F solver, full 12-DOF strain energy computation
│   ├── failure.py             # Combined stress failure criterion, member marking
│   └── redistribution.py      # ODE-based energy transfer after failure
├── entropy/
│   ├── metrics.py             # S, dS/dt, normalized entropy
│   └── localization.py        # Collapse detection (zscore/threshold), Gini index
├── simulation/
│   ├── runner.py              # Main simulation loop
│   └── scenarios.py           # Scenario registry and predefined configs
├── visualization/
│   ├── graph_view.py          # 3D frame viewer with energy heatmap
│   ├── entropy_plot.py        # S, dS/dt, Gini index plots
│   └── animation.py           # GIF/MP4 animation of entropy evolution
├── tests/
│   ├── run_all_tests.py       # Orchestrator for all 7 test phases
│   ├── test_phase1_models.py
│   ├── test_phase2_stiffness.py
│   ├── test_phase3_solver.py
│   ├── test_phase4_failure.py
│   ├── test_phase5_entropy.py
│   ├── test_phase6_simulation.py
│   └── test_phase7_visualization.py
├── main.py                    # CLI entry point
├── benchmark.py               # Validation against analytical + independent solvers
├── pyproject.toml             # Package metadata and build config
├── requirements.txt           # Runtime + build dependencies
└── architecture.ini           # Textual architecture overview (outdated — prefer this file)
```

---

## Architecture Rules

### The Golden Rule: No Cross-Module Imports

Every module communicates **exclusively** through `core/models.py` dataclasses. No module should import internal classes or functions from another module — only from `core/models.py`.

**Correct:**
```python
from core.models import FrameData, SimulationResult
```

**Incorrect:**
```python
from solver.equilibrium import _build_load_vector   # Never do this outside solver/
from structure.stiffness import _local_stiffness     # Never do this outside structure/
```

The only exceptions are the natural parent-child relationships within a layer:
- `main.py` imports from all top-level packages
- `simulation/runner.py` imports from `solver/`, `entropy/`, and `structure/`
- `visualization/` imports from `core/models.py` and `entropy/metrics.py`
- Tests import freely for validation purposes

### Layer Responsibilities

| Layer | Responsibility | Input | Output |
|-------|---------------|-------|--------|
| `core/` | Data contracts | — | `FrameData`, `EnergyState`, `EntropyRecord`, `SimulationResult` |
| `structure/` | Geometry + stiffness | `FrameData` | `K` (global stiffness matrix) |
| `solver/` | Physics | `FrameData` | `EnergyState` |
| `entropy/` | Information theory | `EnergyState` | `EntropyRecord` |
| `simulation/` | Orchestration | `FrameData` | `SimulationResult` |
| `visualization/` | Rendering | `SimulationResult`, `FrameData` | Figures / animations |

---

## Build, Test, and Run Commands

### Installation
```bash
pip install -r requirements.txt
```

### Running the simulator
```bash
# List available scenarios
python main.py --list

# Run a scenario
python main.py --scenario 2d_simple
python main.py --scenario 3d_redundant --method threshold --steps 200
python main.py --scenario pratt_bridge --steps 100 --save

# Produce an animation
python main.py --scenario pratt_bridge --animate --animate-fmt gif
```

**CLI arguments:**
- `--scenario`: Frame to simulate (`2d_simple`, `3d_redundant`, `pratt_bridge`)
- `--method`: Collapse detection — `zscore` (default) or `threshold`
- `--steps`: Maximum simulation steps (default: 100)
- `--load-step`: Load factor increment per step (default: 0.2; 0.0 = static loading)
- `--save`: Save figures to `output_figures/` instead of displaying
- `--animate`: Render energy redistribution animation
- `--animate-fmt`: `gif` (default) or `mp4`
- `--fps`: Animation frames per second (default: 10)

### Testing
```bash
# Run the full 7-phase test suite (29 tests)
python tests/run_all_tests.py

# Run individual phases
python tests/test_phase1_models.py
python tests/test_phase2_stiffness.py
python tests/test_phase3_solver.py
python tests/test_phase4_failure.py
python tests/test_phase5_entropy.py
python tests/test_phase6_simulation.py
python tests/test_phase7_visualization.py
```

Tests are written as standalone scripts using plain `assert` statements. They print `PASS` / `FAIL` per test. The GitHub Actions workflow also attempts `pytest`, but the canonical test runner is `tests/run_all_tests.py`.

### Benchmark validation
```bash
pip install reportlab
python benchmark.py
```

Validates the FEM solver against:
1. Analytical closed-form Euler-Bernoulli (2D beam only)
2. Independent NumPy direct stiffness reimplementation with **no shared code** from `solver/`

Outputs `benchmark_report.pdf` and 8 publication figures.

### Packaging
```bash
python -m build
```

Builds a wheel and sdist from `pyproject.toml`.

### Standalone executable
```bash
pip install pyinstaller
pyinstaller --onefile main.py
```

Output: `dist/main.exe` (Windows) or `dist/main` (Linux/macOS).

---

## Code Style Guidelines

### Docstrings
Every module, class, and function must have a comprehensive docstring explaining:
- What it does
- Args with types and units
- Returns with types and units
- Raises (if applicable)

Docstrings use `"""` triple-double-quotes and follow the Google-style convention seen throughout the codebase.

### Type Hints
Use Python 3.9+ type hint syntax:
```python
def run(frame: FrameData, max_steps: int = 100) -> SimulationResult:
    ...

def _detect(history: list[EntropyRecord], method: str) -> tuple[bool, int | None]:
    ...
```

### Naming Conventions
- `PascalCase` for classes and dataclasses
- `snake_case` for functions, methods, variables, modules
- `UPPER_SNAKE_CASE` for module-level constants
- Private helpers prefixed with `_`

### Units and Scientific Conventions
- Lengths: **meters**
- Forces: **Newtons**
- Stresses: **Pascals**
- Moments: **N·m**
- Energy: **Joules**
- Density: **kg/m³**
- DOF convention: `[0=ux, 1=uy, 2=uz, 3=rx, 4=ry, 5=rz]`

### 2D Frame Constraint Rule
For planar frames in the XY plane, **always** constrain out-of-plane DOFs on all nodes:
```python
PLANAR_DOFS = [2, 3, 4]  # uz, rx, ry
```
This prevents singularity from unconstrained 3D DOFs.

### Boundary Condition Enforcement
Loads applied at supported nodes are **automatically zeroed** during solve. Do not apply loads at pinned/roller support nodes unless you intend to load them before constraints are applied.

---

## Adding New Frames

1. Create `structure/frames/frame_<name>.py`
2. Define materials using `dataclasses.replace(STEEL_S275, ...)` or a custom `Material`
3. Implement `build() -> FrameData` with nodes, members, loads
4. For 2D frames: add `fixed_dofs=[2, 3, 4]` to all nodes
5. Register in `simulation/scenarios.py` and `main.py`

---

## Testing Philosophy

The test suite is organized in **7 phases** that mirror the architecture layers:

| Phase | File | Coverage |
|-------|------|----------|
| 1 | `test_phase1_models.py` | Dataclass instantiation, Material properties, frame `build()` |
| 2 | `test_phase2_stiffness.py` | `K` shape, symmetry, boundary condition enforcement |
| 3 | `test_phase3_solver.py` | Displacement correctness, strain energy magnitude, 3D frame |
| 4 | `test_phase4_failure.py` | Combined stress criterion, member flag, energy conservation |
| 5 | `test_phase5_entropy.py` | `S` formula, `dS` sign, normalized entropy, Gini index |
| 6 | `test_phase6_simulation.py` | End-to-end runs, collapse detection, failure sequence order |
| 7 | `test_phase7_visualization.py` | Plot functions save without error, collapse overlay renders |

When adding new features, add tests to the appropriate phase file. Keep tests deterministic (no random data without fixed seeds).

---

## CI/CD

Two GitHub Actions workflows are defined in `.github/workflows/`:

1. **`python-package.yml`**: Runs on push/PR to `main`
   - Tests against Python 3.9, 3.10, 3.11
   - Lints with `flake8` (syntax errors are fatal; style issues are warnings)
   - Runs `pytest`

2. **`python-publish.yml`**: Triggers on release publication
   - Builds package with `python -m build`
   - Publishes to PyPI using trusted publishing (no API token)

**Note:** The canonical test runner is `python tests/run_all_tests.py`. The `pytest` invocation in CI may discover tests, but the standalone scripts are the ground truth.

---

## Security Considerations

This is a pure scientific/research desktop application. It does not:
- Expose network services
- Handle user authentication or authorization
- Process untrusted user input beyond CLI arguments
- Store sensitive data

When modifying code, maintain the principle that the simulator operates only on local data files and CLI arguments. Avoid introducing network dependencies or executable string evaluation (`eval`, `exec`) unless explicitly justified.

---

## Key Configuration Files

| File | Purpose |
|------|---------|
| `pyproject.toml` | Package metadata, dependencies, build backend, console script entry point |
| `requirements.txt` | Runtime dependencies + PyInstaller (for executable builds) |
| `architecture.ini` | Legacy textual architecture diagram (may be outdated) |
| `.github/workflows/python-package.yml` | CI: test + lint on push/PR |
| `.github/workflows/python-publish.yml` | CD: publish to PyPI on release |

---

## Important Behavioral Notes

- **Linear elastic only** — no plasticity, catenary action, or geometric nonlinearity.
- **Quasi-static** — no dynamic amplification (no DAF).
- **Failure criterion:** `sigma_max = |N|/A + |M_max| * c / I` compared against `sigma_y`.
- **Energy redistribution** is phenomenological (ODE coupling based on shared-node stiffness), not derived from equilibrium.
- **Z-score collapse detection** is recommended for research; **threshold** is simpler but requires per-frame calibration.
- When `load_factor_step=0.0`, the simulation runs at the design load with no incremental ramp.

---

## Commit Conventions

This project uses **Conventional Commits** (`https://www.conventionalcommits.org/`):

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

**Common types:**

| Type | Use when |
|------|----------|
| `feat` | New feature or frame scenario |
| `fix` | Bug fix in solver, entropy logic, or visualization |
| `docs` | README, AGENTS.md, docstring updates only |
| `style` | Formatting, whitespace, trailing commas |
| `refactor` | Code change that neither fixes a bug nor adds a feature |
| `test` | Adding or correcting tests |
| `chore` | Build, CI, dependency bumps |

**Examples:**
```
feat(structure): add warren truss frame scenario
fix(solver): correct strain energy sign for compression members
test(entropy): add deterministic seed for gini index test
```

---

## Agent Interaction Standards

### File Delivery
- **Present files one at a time** — wait for feedback before the next file.
- **Fixes and improvements:** diffs / snippets only — never full files unless explicitly asked.
- **Never volunteer a full file** when a targeted change is sufficient.

### Code Style (beyond project rules)
- **Comments:** 1-line only — no multi-line or block comments anywhere.
- **Bug fixes:** always at the root cause — never patch test parameters or create workarounds to produce passing results. If a test fails because the physics is wrong, fix the physics.
- **Never write code just to make it compile** — code must reflect real behavior.

### Communication Style
- Brief and factually correct — no over-explaining simple things.
- No bullet points for conversational answers — prose only.
- No emojis unless used first.
- When asked for a recommendation, give one — don't hedge with multiple options.
- If something needs research before answering, search the web first — don't guess.

### Environment
- **IDE:** VS Code
- **Terminal:** PowerShell (Windows) — never use `&&` separator, always separate commands.
- **Python:** Always create and activate a venv before installing dependencies — multiple conflicting Python installs exist on this machine via PyManager (system + user). Never skip this step.

### Session Handoff
At the end of each project session, the agent should:
1. Update the project-specific `memory.md` with current status.
2. Note any v.next priorities in order.
3. Note any pending items that were deferred.

---

*Last updated: 2026-05-03*
