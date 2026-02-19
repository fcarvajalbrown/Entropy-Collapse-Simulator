"""
simulation/scenarios.py
=======================
Predefined simulation scenarios for quick testing and validation.

Each scenario loads a frame, configures runner parameters, and returns
a SimulationResult. New scenarios can be added by following the same
pattern — one function per scenario, all returning SimulationResult.

To add a new scenario:
  1. Add a frame file to structure/frames/
  2. Define a function here that calls runner.run() with desired config
  3. Register it in SCENARIOS dict at the bottom of this file
"""

from core.models import SimulationResult
from simulation import runner
from structure.frames import frame_2d_simple, frame_3d_redundant, frame_pratt_bridge


# ---------------------------------------------------------------------------
# Scenario definitions
# ---------------------------------------------------------------------------

def scenario_2d_simple(
    max_steps: int = 100,
    collapse_method: str = "zscore"
) -> SimulationResult:
    """
    Basic 2D simply-supported frame under a central point load.

    Uses incremental loading (load_factor_step=0.5) so member failures
    and entropy evolution occur within a reasonable step count.

    Args:
        max_steps: Maximum simulation steps.
        collapse_method: "zscore" or "threshold".

    Returns:
        SimulationResult from the runner.
    """
    frame = frame_2d_simple.build()
    return runner.run(
        frame,
        max_steps=max_steps,
        collapse_method=collapse_method,
        load_factor_start=1.0,
        load_factor_step=0.5,
    )


def scenario_3d_redundant(
    max_steps: int = 150,
    collapse_method: str = "zscore"
) -> SimulationResult:
    """
    3D redundant space frame — tests energy redistribution across
    multiple load paths after member failures.

    Uses incremental loading (load_factor_step=0.3) to drive progressive
    failures across the redundant load paths.

    Args:
        max_steps: Maximum simulation steps.
        collapse_method: "zscore" or "threshold".

    Returns:
        SimulationResult from the runner.
    """
    frame = frame_3d_redundant.build()
    return runner.run(
        frame,
        max_steps=max_steps,
        collapse_method=collapse_method,
        load_factor_start=1.0,
        load_factor_step=0.3,
    )


def scenario_pratt_bridge(
    max_steps: int = 200,
    collapse_method: str = "zscore"
) -> SimulationResult:
    """
    6-panel Pratt truss bridge under distributed traffic loading.

    Uses incremental loading (load_factor_step=0.05) to simulate
    gradually increasing traffic load until progressive member failures
    and eventual collapse.

    Args:
        max_steps: Maximum simulation steps.
        collapse_method: "zscore" or "threshold".

    Returns:
        SimulationResult from the runner.
    """
    frame = frame_pratt_bridge.build()
    return runner.run(
        frame,
        max_steps=max_steps,
        collapse_method=collapse_method,
        load_factor_start=1.0,
        load_factor_step=0.2,
    )


# ---------------------------------------------------------------------------
# Scenario registry
# ---------------------------------------------------------------------------

SCENARIOS: dict[str, callable] = {
    "2d_simple":     scenario_2d_simple,
    "3d_redundant":  scenario_3d_redundant,
    "pratt_bridge":  scenario_pratt_bridge,
}


def run_scenario(name: str, **kwargs) -> SimulationResult:
    """
    Run a scenario by name with optional keyword overrides.

    Args:
        name: Scenario key from SCENARIOS registry.
        **kwargs: Passed directly to the scenario function
                  (e.g. max_steps=200, collapse_method="threshold").

    Returns:
        SimulationResult from the selected scenario.

    Raises:
        ValueError: If scenario name is not found in registry.
    """
    if name not in SCENARIOS:
        available = ", ".join(SCENARIOS.keys())
        raise ValueError(f"Unknown scenario '{name}'. Available: {available}")
    return SCENARIOS[name](**kwargs)


def list_scenarios() -> list[str]:
    """Return all registered scenario names."""
    return list(SCENARIOS.keys())