"""
manuscript/results.py
=====================
Single source of every COMPUTED number used by the per-journal manuscript
build scripts under manuscript/<JOURNAL>/. Each builder imports
compute_results() from here, so the numbers reported in every journal version
are produced from the live analysis modules and can never drift out of sync
with the code.
"""

import os
import sys

# The analysis modules live at the project root (one level above manuscript/).
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from structure.frames import (
    frame_building_2d, frame_pratt_bridge, frame_2d_simple, frame_vogel_six_storey
)
from entropy import robustness as rb
from analysis import criteria as C
from analysis import importance as imp
from analysis import parametric as par
from analysis import design_variants as dv
import benchmark as bench

TRUSS_STEP = 0.3


def compute_results() -> dict:
    beam = rb.analyze(frame_2d_simple.build())
    frame = rb.analyze(frame_building_2d.build())
    truss = rb.analyze(frame_pratt_bridge.build())
    large = rb.analyze(frame_building_2d.build(n_bays=3, n_stories=6))
    large_frame = frame_building_2d.build(n_bays=3, n_stories=6)

    cmp = C.compare(frame_pratt_bridge.build(), max_steps=40, load_factor_step=TRUSS_STEP)
    triggers = {k: (cmp.history[v].load_factor if v is not None else None)
                for k, v in cmp.first_trigger.items()}

    imp_frame = imp.compare(frame_building_2d.build())
    imp_truss = imp.compare(frame_pratt_bridge.build())
    sweep = par.redundancy_sweep()
    ssens = par.step_size_sensitivity()
    ensemble = par.importance_ensemble()

    analytical = bench.run_analytical()
    independent = bench.run_independent()
    index_checks = bench.run_index_validation()

    return dict(beam=beam, frame=frame, truss=truss, large=large,
                large_n=(len(large_frame.nodes), len(large_frame.members)),
                triggers=triggers, imp_frame=imp_frame, imp_truss=imp_truss,
                sweep=sweep, ssens=ssens, ensemble=ensemble,
                analytical=analytical, independent=independent,
                index_checks=index_checks)


def compute_paper2_results() -> dict:
    """
    Live numbers for paper 2 (the construction-journal screening-tool extension):
    the Vogel (1985) benchmark building, the R_S-vs-code-ALP column agreement,
    and the base-fixity design-variant study. Kept in the same shared module so
    the paper-2 build script cannot drift from the code either.
    """
    vogel_frame = frame_vogel_six_storey.build()
    vogel = rb.analyze(vogel_frame)
    n_nodes, n_members = len(vogel_frame.nodes), len(vogel_frame.members)
    n_columns = len(imp.column_member_ids(vogel_frame))

    alp = imp.compare_column_alp(vogel_frame)
    variants = dv.vogel_base_fixity_study()

    # Independent dual-solver cross-check row for the Vogel frame (benchmark.py).
    independent = bench.run_independent()
    vogel_indep = next((r for r in independent
                        if "Vogel" in r["frame"]), None)

    return dict(
        vogel=vogel,
        vogel_n=(n_nodes, n_members, n_columns),
        alp=alp,
        variants=variants,
        vogel_indep=vogel_indep,
    )
