"""
benchmark.py
============
Verification and validation harness for the planar frame solver.

The solver is validated at two independent tiers, neither of which shares any
code with structure/ or solver/:

  Tier 1 - Analytical (exact).  Three Euler-Bernoulli cases with closed-form
           solutions (simply-supported, fixed-fixed and cantilever beams).
           The displacement-method frame element is exact for point loads
           applied at nodes, so the expected error is ~0 (floating point).

  Tier 2 - Independent dual solver.  A from-scratch 3-DOF-per-node planar
           frame solver (independent_solve, below) re-analyses the redundant
           benchmark frames; its joint displacements and total strain energy
           are compared against the production solver.

Note on external benchmarks: the Ziemian & Ziemian steel benchmark frames
(Data in Brief, https://doi.org/10.1016/j.dib.2021.107510;
data: https://doi.org/10.17632/39sjhchwtx.1) are a recognized external
reference. Their verified results are SECOND-ORDER (P-Delta / stability),
which this first-order linear solver does not model; they are cited as a
recommended future cross-check rather than reproduced here, to avoid
comparing against results outside this solver's modelling scope.

Run:
    python benchmark.py            # prints tables, writes validation/ report
    python benchmark.py --figures  # additionally saves comparison figures
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from core.models import FrameData, Node, Member, Load, Material
from solver.equilibrium import solve_full
from structure.frames import frame_building_2d, frame_pratt_bridge


# ===========================================================================
# Independent 3-DOF planar frame solver (no shared code with structure/solver)
# ===========================================================================

def _planar_element_stiffness(E, A, I, L) -> np.ndarray:
    """6x6 local stiffness of a 2D Euler-Bernoulli frame element."""
    k = np.zeros((6, 6))
    ea = E * A / L
    ei = E * I
    k[0, 0] = k[3, 3] = ea
    k[0, 3] = k[3, 0] = -ea
    k[1, 1] = k[4, 4] = 12 * ei / L**3
    k[1, 4] = k[4, 1] = -12 * ei / L**3
    k[1, 2] = k[2, 1] = 6 * ei / L**2
    k[1, 5] = k[5, 1] = 6 * ei / L**2
    k[4, 2] = k[2, 4] = -6 * ei / L**2
    k[4, 5] = k[5, 4] = -6 * ei / L**2
    k[2, 2] = k[5, 5] = 4 * ei / L
    k[2, 5] = k[5, 2] = 2 * ei / L
    return k


def _planar_transform(c, s) -> np.ndarray:
    """6x6 transformation matrix from global to local for direction (c, s)."""
    T = np.zeros((6, 6))
    R = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])
    T[:3, :3] = R
    T[3:, 3:] = R
    return T


def independent_solve(frame: FrameData) -> Tuple[Dict[int, np.ndarray], float]:
    """
    Solve a planar frame with a self-contained 3-DOF-per-node direct stiffness
    method. Reads geometry/loads from a FrameData but shares NO code with the
    production assembler/solver.

    In-plane DOFs only: (ux, uy, rz) -> local indices (0, 1, 2). Out-of-plane
    DOFs in the FrameData are ignored. A node DOF is fixed if it appears in
    fixed_dofs as 0 (ux), 1 (uy) or 5 (rz).

    Returns:
        (displacements, total_strain_energy)
        displacements maps node_id -> array([ux, uy, rz]).
    """
    node_index = {n.id: i for i, n in enumerate(frame.nodes)}
    coords = {n.id: (n.x, n.y) for n in frame.nodes}
    n_dof = 3 * len(frame.nodes)
    K = np.zeros((n_dof, n_dof))

    def gdof(node_id, local):
        return 3 * node_index[node_id] + local

    for m in frame.members:
        if m.failed:
            continue
        (x1, y1), (x2, y2) = coords[m.node_start], coords[m.node_end]
        L = math.hypot(x2 - x1, y2 - y1)
        c, s = (x2 - x1) / L, (y2 - y1) / L
        # Truss bars carry axial force only; zero the bending part of I.
        I_eff = 0.0 if getattr(m, "is_truss", False) else m.I
        kl = _planar_element_stiffness(m.E, m.A, I_eff, L)
        T = _planar_transform(c, s)
        kg = T.T @ kl @ T
        dofs = [gdof(m.node_start, 0), gdof(m.node_start, 1), gdof(m.node_start, 2),
                gdof(m.node_end, 0), gdof(m.node_end, 1), gdof(m.node_end, 2)]
        for a in range(6):
            for b in range(6):
                K[dofs[a], dofs[b]] += kg[a, b]

    F = np.zeros(n_dof)
    local_of = {0: 0, 1: 1, 5: 2}
    for load in frame.loads:
        if load.dof in local_of:
            F[gdof(load.node_id, local_of[load.dof])] += load.magnitude

    # Reduce out fixed DOFs.
    fixed = set()
    for n in frame.nodes:
        for d in n.fixed_dofs:
            if d in local_of:
                fixed.add(gdof(n.id, local_of[d]))
    free = [i for i in range(n_dof) if i not in fixed]

    u = np.zeros(n_dof)
    Kff = K[np.ix_(free, free)]
    u[free] = np.linalg.solve(Kff, F[free])

    total_energy = float(0.5 * u @ K @ u)

    displacements = {
        n.id: u[3 * node_index[n.id]: 3 * node_index[n.id] + 3] for n in frame.nodes
    }
    return displacements, total_energy


def independent_member_energy(frame: FrameData) -> Dict[int, float]:
    """
    Per-member elastic strain energy from the independent solver, used to
    cross-check the strain-energy DISTRIBUTION {p_i} that drives the entropy.
    """
    disp, _ = independent_solve(frame)
    coords = {n.id: (n.x, n.y) for n in frame.nodes}
    out: Dict[int, float] = {}
    for m in frame.members:
        if m.failed:
            out[m.id] = 0.0
            continue
        (x1, y1), (x2, y2) = coords[m.node_start], coords[m.node_end]
        L = math.hypot(x2 - x1, y2 - y1)
        c, s = (x2 - x1) / L, (y2 - y1) / L
        I_eff = 0.0 if getattr(m, "is_truss", False) else m.I
        kl = _planar_element_stiffness(m.E, m.A, I_eff, L)
        T = _planar_transform(c, s)
        ue = np.concatenate([disp[m.node_start], disp[m.node_end]])  # 6 in global
        ul = T @ ue
        out[m.id] = float(0.5 * ul @ kl @ ul)
    return out


# ===========================================================================
# Tier 1 - Analytical closed-form cases
# ===========================================================================

@dataclass
class AnalyticalCase:
    name: str
    frame: FrameData
    probe_node: int          # node whose vertical deflection is checked
    delta_exact: float       # closed-form deflection magnitude (m)
    U_exact: float           # closed-form strain energy (J)
    reference: str


def _beam_material(E=200e9, A=0.01, I=1e-4) -> Material:
    return Material(name="benchmark", E=E, A=A, I=I, sigma_y=1e30, c=0.1)


def _simply_supported_case() -> AnalyticalCase:
    """Simply-supported beam, central point load. delta = PL^3/48EI."""
    E, I, A = 200e9, 1e-4, 0.01
    L, P = 10.0, 50_000.0
    mat = _beam_material(E, A, I)
    nodes = [
        Node(0, 0.0, 0.0, 0.0, fixed_dofs=[0, 1, 2, 3, 4]),     # pin (rz free)
        Node(1, L / 2, 0.0, 0.0, fixed_dofs=[2, 3, 4]),
        Node(2, L, 0.0, 0.0, fixed_dofs=[1, 2, 3, 4]),          # roller (ux free, rz free)
    ]
    members = [Member(0, 0, 1, mat), Member(1, 1, 2, mat)]
    loads = [Load(1, 1, -P)]
    frame = FrameData("Simply-supported beam", nodes, members, loads)
    delta = P * L**3 / (48 * E * I)
    U = P**2 * L**3 / (96 * E * I)
    return AnalyticalCase("Simply-supported beam", frame, 1, delta, U,
                          "Gere & Goodno, Mechanics of Materials")


def _fixed_fixed_case() -> AnalyticalCase:
    """Clamped-clamped beam, central point load. delta = PL^3/192EI."""
    E, I, A = 200e9, 1e-4, 0.01
    L, P = 10.0, 50_000.0
    mat = _beam_material(E, A, I)
    nodes = [
        Node(0, 0.0, 0.0, 0.0, fixed_dofs=[0, 1, 2, 3, 4, 5]),  # fully fixed
        Node(1, L / 2, 0.0, 0.0, fixed_dofs=[2, 3, 4]),
        Node(2, L, 0.0, 0.0, fixed_dofs=[0, 1, 2, 3, 4, 5]),    # fully fixed
    ]
    members = [Member(0, 0, 1, mat), Member(1, 1, 2, mat)]
    loads = [Load(1, 1, -P)]
    frame = FrameData("Fixed-fixed beam", nodes, members, loads)
    delta = P * L**3 / (192 * E * I)
    U = P**2 * L**3 / (384 * E * I)
    return AnalyticalCase("Fixed-fixed beam", frame, 1, delta, U,
                          "Roark's Formulas for Stress and Strain")


def _cantilever_case() -> AnalyticalCase:
    """Cantilever, end point load. delta = PL^3/3EI."""
    E, I, A = 200e9, 1e-4, 0.01
    L, P = 5.0, 20_000.0
    mat = _beam_material(E, A, I)
    nodes = [
        Node(0, 0.0, 0.0, 0.0, fixed_dofs=[0, 1, 2, 3, 4, 5]),  # fully fixed
        Node(1, L, 0.0, 0.0, fixed_dofs=[2, 3, 4]),
    ]
    members = [Member(0, 0, 1, mat)]
    loads = [Load(1, 1, -P)]
    frame = FrameData("Cantilever beam", nodes, members, loads)
    delta = P * L**3 / (3 * E * I)
    U = P**2 * L**3 / (6 * E * I)
    return AnalyticalCase("Cantilever beam", frame, 1, delta, U,
                          "Gere & Goodno, Mechanics of Materials")


def run_analytical() -> List[dict]:
    """Run the three analytical cases and return per-case error rows."""
    cases = [_simply_supported_case(), _fixed_fixed_case(), _cantilever_case()]
    rows = []
    for case in cases:
        u, energy_state = solve_full(case.frame, step=0)
        delta_num = abs(u[case.probe_node * 6 + 1])  # vertical (uy) at probe
        U_num = energy_state.total_energy
        rows.append({
            "case": case.name,
            "delta_exact": case.delta_exact,
            "delta_num": delta_num,
            "delta_err_pct": _pct_err(delta_num, case.delta_exact),
            "U_exact": case.U_exact,
            "U_num": U_num,
            "U_err_pct": _pct_err(U_num, case.U_exact),
            "reference": case.reference,
        })
    return rows


# ===========================================================================
# Tier 2 - Independent dual solver cross-check
# ===========================================================================

def run_independent() -> List[dict]:
    """Cross-check the production solver against the independent solver."""
    rows = []
    builders = [
        frame_building_2d.build,
        lambda: frame_building_2d.build(n_bays=3, n_stories=6),  # larger case study
        frame_pratt_bridge.build,
    ]
    for builder in builders:
        frame = builder()
        u_prod, es = solve_full(frame, step=0)
        disp_ind, U_ind = independent_solve(frame)

        # Peak in-plane translation from each solver.
        peak_prod = max(
            math.hypot(u_prod[n.id * 6 + 0], u_prod[n.id * 6 + 1])
            for n in frame.nodes
        )
        peak_ind = max(math.hypot(d[0], d[1]) for d in disp_ind.values())

        rows.append({
            "frame": frame.name,
            "peak_disp_prod": peak_prod,
            "peak_disp_ind": peak_ind,
            "disp_err_pct": _pct_err(peak_prod, peak_ind),
            "U_prod": es.total_energy,
            "U_ind": U_ind,
            "U_err_pct": _pct_err(es.total_energy, U_ind),
        })
    return rows


# ===========================================================================
# Tier 3 - Index validation (the quantities R_S actually depends on)
# ===========================================================================
# The displacement/energy tiers above verify the intact solver. These checks
# verify the per-member distribution {p_i}, the post-removal ALP states, the
# failure criterion, and the stability (mechanism) test - i.e. the inputs to
# the Entropy Robustness Index, addressing the concern that "0% on intact
# totals" does not by itself validate R_S.

def _distribution(member_energy: Dict[int, float]) -> Dict[int, float]:
    total = sum(member_energy.values())
    if total <= 0:
        return {k: 0.0 for k in member_energy}
    return {k: v / total for k, v in member_energy.items()}


def run_index_validation() -> List[dict]:
    """Cross-checks of the distribution, ALP states, failure criterion, stability."""
    import copy
    from solver.equilibrium import solve
    from solver.failure import _combined_stress
    from solver.equilibrium import solve_full as _solve_full
    from entropy.robustness import is_stable
    from core.models import FrameData, Node, Member, Load
    from structure.frames import frame_building_2d, frame_pratt_bridge

    rows: List[dict] = []

    # (a) Per-member distribution {p_i}: production vs independent solver.
    for mod in (frame_building_2d, frame_pratt_bridge):
        frame = mod.build()
        prod = solve(frame, step=0)
        p_prod = _distribution({ms.member_id: ms.strain_energy for ms in prod.member_states})
        p_ind = _distribution(independent_member_energy(frame))
        max_dp = max(abs(p_prod[k] - p_ind[k]) for k in p_prod)
        rows.append({"check": f"per-member p_i  [{frame.name[:28]}]",
                     "metric": "max |dp_i|", "value": f"{max_dp:.2e}",
                     "pass": max_dp < 1e-6})

    # (b) Post-removal ALP state: remove the critical member, re-solve both.
    for mod in (frame_building_2d, frame_pratt_bridge):
        frame = mod.build()
        from entropy.robustness import analyze
        crit = analyze(frame).critical_member
        work = copy.deepcopy(frame)
        next(m for m in work.members if m.id == crit).failed = True
        u_prod, es = _solve_full(work, step=0)
        _, U_ind = independent_solve(work)
        err = _pct_err(es.total_energy, U_ind)
        rows.append({"check": f"post-removal ALP (rm {crit}) [{frame.name[:18]}]",
                     "metric": "U err %", "value": f"{err:.4f}",
                     "pass": err < 1e-3})

    # (c) Failure criterion by hand: cantilever, end transverse load P.
    #     N = 0, M_max = P*L at the fixed end, sigma = M*c/I.
    E, A, I, c = 200e9, 0.01, 1e-4, 0.1
    L, P = 5.0, 20_000.0
    mat = _beam_material(E, A, I)
    f = FrameData("fail-check",
                  [Node(0, 0.0, 0.0, 0.0, fixed_dofs=[0, 1, 2, 3, 4, 5]),
                   Node(1, L, 0.0, 0.0, fixed_dofs=[2, 3, 4])],
                  [Member(0, 0, 1, mat)], [Load(1, 1, -P)])
    u, _ = _solve_full(f, step=0)
    sigma_code = _combined_stress(f.members[0], u, f)
    sigma_hand = (P * L) * c / I
    err = _pct_err(sigma_code, sigma_hand)
    rows.append({"check": "failure criterion (cantilever M*c/I)",
                 "metric": "stress err %", "value": f"{err:.4f}",
                 "pass": err < 1e-6})

    # (d) Stability test: known mechanism vs known stable removal.
    beam = frame_building_2d.build()
    stable_after_beam = is_stable(_with_failed(beam, beam.members[-1].id))   # remove a beam
    det_truss = frame_pratt_bridge.build(n_counter=0)                        # determinate
    mech_after_truss = not is_stable(_with_failed(det_truss, 0))             # remove a chord
    rows.append({"check": "stability: stable removal stays stable",
                 "metric": "is_stable", "value": str(stable_after_beam),
                 "pass": stable_after_beam})
    rows.append({"check": "stability: determinate-truss removal = mechanism",
                 "metric": "mechanism", "value": str(mech_after_truss),
                 "pass": mech_after_truss})
    return rows


def _with_failed(frame, member_id):
    import copy
    work = copy.deepcopy(frame)
    next(m for m in work.members if m.id == member_id).failed = True
    return work


# ===========================================================================
# Reporting
# ===========================================================================

def _pct_err(num, ref) -> float:
    if ref == 0:
        return 0.0 if num == 0 else float("inf")
    return abs(num - ref) / abs(ref) * 100.0


def _write_report(analytical: List[dict], independent: List[dict],
                  index_checks: List[dict], path: str) -> None:
    lines = ["# Solver validation report", ""]
    lines.append("## Tier 1 - Analytical (exact closed-form)")
    lines.append("")
    lines.append("| Case | delta_exact (m) | delta_num (m) | err % | U_exact (J) | U_num (J) | err % | Reference |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in analytical:
        lines.append(
            f"| {r['case']} | {r['delta_exact']:.6e} | {r['delta_num']:.6e} | "
            f"{r['delta_err_pct']:.4f} | {r['U_exact']:.6e} | {r['U_num']:.6e} | "
            f"{r['U_err_pct']:.4f} | {r['reference']} |"
        )
    lines.append("")
    lines.append("## Tier 2 - Independent dual solver cross-check")
    lines.append("")
    lines.append("| Frame | peak disp (production, m) | peak disp (independent, m) | err % | U production (J) | U independent (J) | err % |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in independent:
        lines.append(
            f"| {r['frame']} | {r['peak_disp_prod']:.6e} | {r['peak_disp_ind']:.6e} | "
            f"{r['disp_err_pct']:.4f} | {r['U_prod']:.6e} | {r['U_ind']:.6e} | "
            f"{r['U_err_pct']:.4f} |"
        )
    lines.append("")
    lines.append("## Tier 3 - Index validation (distribution, ALP, failure, stability)")
    lines.append("")
    lines.append("| Check | Metric | Value | Pass |")
    lines.append("|---|---|---|---|")
    for r in index_checks:
        lines.append(f"| {r['check']} | {r['metric']} | {r['value']} | "
                     f"{'yes' if r['pass'] else 'NO'} |")
    lines.append("")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _print_rows(analytical: List[dict], independent: List[dict]) -> None:
    print("\n=== Tier 1: Analytical (exact) ===")
    print(f"{'Case':<24}{'delta err %':>14}{'U err %':>12}")
    for r in analytical:
        print(f"{r['case']:<24}{r['delta_err_pct']:>14.4f}{r['U_err_pct']:>12.4f}")
    print("\n=== Tier 2: Independent dual solver ===")
    print(f"{'Frame':<40}{'disp err %':>12}{'U err %':>12}")
    for r in independent:
        print(f"{r['frame']:<40}{r['disp_err_pct']:>12.4f}{r['U_err_pct']:>12.4f}")


def _print_index(index_checks: List[dict]) -> None:
    print("\n=== Tier 3: Index validation (distribution / ALP / failure / stability) ===")
    for r in index_checks:
        flag = "PASS" if r["pass"] else "FAIL"
        print(f"  [{flag}] {r['check']:<46} {r['metric']}={r['value']}")


def _save_figures(analytical: List[dict], independent: List[dict], out_dir: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"(matplotlib unavailable, skipping figures: {exc})")
        return

    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    labels = [r["case"] for r in analytical] + [r["frame"] for r in independent]
    errs = [r["delta_err_pct"] for r in analytical] + [r["disp_err_pct"] for r in independent]
    colors = ["steelblue"] * len(analytical) + ["firebrick"] * len(independent)
    ax.bar(range(len(labels)), errs, color=colors)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Displacement error (%)")
    ax.set_title("Solver validation: analytical (blue) and independent solver (red)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "validation_errors.png"), dpi=150)
    plt.close(fig)
    print(f"Saved figure: {os.path.join(out_dir, 'validation_errors.png')}")


def main():
    parser = argparse.ArgumentParser(description="Solver validation harness")
    parser.add_argument("--figures", action="store_true",
                        help="Save comparison figures to output_figures/")
    args = parser.parse_args()

    analytical = run_analytical()
    independent = run_independent()
    index_checks = run_index_validation()

    _print_rows(analytical, independent)
    _print_index(index_checks)
    report_path = os.path.join("validation", "validation_report.md")
    _write_report(analytical, independent, index_checks, report_path)
    print(f"\nWrote {report_path}")

    if args.figures:
        _save_figures(analytical, independent, "output_figures")


if __name__ == "__main__":
    main()
