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
from structure.frames import (
    frame_2d_simple, frame_building_2d, frame_pratt_bridge, frame_vogel_six_storey
)


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


def scenario_building_2d(
    max_steps: int = 150,
    collapse_method: str = "zscore"
) -> SimulationResult:
    """
    Planar 2-bay, 3-story steel moment frame — a redundant building frame
    that redistributes load through frame action after member failures.

    Uses incremental loading (load_factor_step=0.3) to drive progressive
    failures across the redundant load paths.

    Args:
        max_steps: Maximum simulation steps.
        collapse_method: "zscore" or "threshold".

    Returns:
        SimulationResult from the runner.
    """
    frame = frame_building_2d.build()
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


def scenario_vogel_six_storey(
    max_steps: int = 150,
    collapse_method: str = "zscore"
) -> SimulationResult:
    """
    Vogel (1985) six-storey two-bay steel calibration frame -- a realistic,
    published-benchmark redundant moment frame (see
    structure/frames/frame_vogel_six_storey.py for provenance and scope).

    Uses incremental loading (load_factor_step=0.3) to drive progressive
    failures across the redundant load paths.

    Args:
        max_steps: Maximum simulation steps.
        collapse_method: "zscore" or "threshold".

    Returns:
        SimulationResult from the runner.
    """
    frame = frame_vogel_six_storey.build()
    return runner.run(
        frame,
        max_steps=max_steps,
        collapse_method=collapse_method,
        load_factor_start=1.0,
        load_factor_step=0.3,
    )


# ---------------------------------------------------------------------------
# Scenario registry
# ---------------------------------------------------------------------------

SCENARIOS: dict[str, callable] = {
    "2d_simple":        scenario_2d_simple,
    "building_2d":      scenario_building_2d,
    "pratt_bridge":     scenario_pratt_bridge,
    "vogel_six_storey": scenario_vogel_six_storey,
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