"""
solver/redistribution.py
========================
Models energy redistribution between members after a failure event.

When a member fails, its stored strain energy does not vanish — it
redistributes to connected members based on stiffness connectivity.

The governing equation is:
    dU_i/dt = sum_j [ alpha_ij * (U_j - U_i) ]

where alpha_ij is the coupling coefficient between members i and j,
derived from shared node connectivity and relative stiffness.

This is solved as a discrete time step using forward Euler integration.
Consumed by simulation/runner.py immediately after a failure event.
"""

import numpy as np
from core.models import FrameData, EnergyState, MemberState


def redistribute(frame: FrameData, energy_state: EnergyState, dt: float = 1.0) -> EnergyState:
    """
    Perform one redistribution step after a failure event.

    Builds the coupling matrix alpha from frame topology, then advances
    the energy state by dt using forward Euler integration.

    Args:
        frame: Current frame (failed members excluded from coupling).
        energy_state: Energy state immediately after failure marking.
        dt: Integration time step. Default 1.0 (normalized pseudo-time).

    Returns:
        Updated EnergyState with redistributed strain energies.
    """
    active = [ms for ms in energy_state.member_states if not ms.failed]
    ids = [ms.member_id for ms in active]
    U = np.array([ms.strain_energy for ms in active], dtype=float)

    alpha = _build_coupling_matrix(frame, ids)
    dU = alpha @ U - np.diag(alpha.sum(axis=1)) @ U
    U_new = np.clip(U + dt * dU, 0.0, None)  # Energy cannot go negative

    updated_map = dict(zip(ids, U_new))

    new_member_states = []
    for ms in energy_state.member_states:
        if ms.member_id in updated_map:
            new_member_states.append(MemberState(
                member_id=ms.member_id,
                strain_energy=updated_map[ms.member_id],
                axial_force=ms.axial_force,
                deformation=ms.deformation,
                failed=ms.failed
            ))
        else:
            new_member_states.append(ms)  # Failed member, unchanged

    total_energy = sum(ms.strain_energy for ms in new_member_states)

    return EnergyState(
        step=energy_state.step,
        total_energy=total_energy,
        member_states=new_member_states
    )


def _build_coupling_matrix(frame: FrameData, active_ids: list[int]) -> np.ndarray:
    """
    Build the coupling coefficient matrix alpha for active members.

    Two members are coupled if they share a node. The coupling strength
    alpha_ij is proportional to the harmonic mean of their axial stiffnesses
    (EA/L), normalized so that stiffer paths attract more energy.

    Args:
        frame: Frame definition for topology and member properties.
        active_ids: IDs of non-failed members to include.

    Returns:
        alpha (np.ndarray): Square coupling matrix of shape (n, n).
                            alpha[i, j] > 0 if members i and j share a node.
                            Diagonal is zero (no self-coupling).
    """
    n = len(active_ids)
    alpha = np.zeros((n, n))
    idx = {mid: i for i, mid in enumerate(active_ids)}

    active_members = [m for m in frame.members if m.id in idx]

    # Build node -> member adjacency
    node_members: dict[int, list] = {}
    for m in active_members:
        for node_id in (m.node_start, m.node_end):
            node_members.setdefault(node_id, []).append(m)

    # Fill coupling coefficients for members sharing a node
    for members_at_node in node_members.values():
        for m_i in members_at_node:
            for m_j in members_at_node:
                if m_i.id == m_j.id:
                    continue
                i, j = idx[m_i.id], idx[m_j.id]
                k_i = _axial_stiffness(m_i, frame)
                k_j = _axial_stiffness(m_j, frame)
                alpha[i, j] += _harmonic_mean(k_i, k_j)

    return alpha


def _axial_stiffness(member, frame: FrameData) -> float:
    """
    Compute axial stiffness EA/L for a member.

    Args:
        member: Member with E, A properties.
        frame: Used to compute member length from node coordinates.

    Returns:
        Axial stiffness in N/m.
    """
    n_start = _get_node(frame, member.node_start)
    n_end = _get_node(frame, member.node_end)
    L = np.sqrt(
        (n_end.x - n_start.x)**2 +
        (n_end.y - n_start.y)**2 +
        (n_end.z - n_start.z)**2
    )
    return member.E * member.A / L


def _harmonic_mean(a: float, b: float) -> float:
    """
    Compute harmonic mean of two stiffness values.
    Returns 0 if either value is zero to avoid division errors.

    Args:
        a, b: Stiffness values.

    Returns:
        Harmonic mean, or 0.0 if inputs are invalid.
    """
    if a == 0 or b == 0:
        return 0.0
    return 2 * a * b / (a + b)


def _get_node(frame: FrameData, node_id: int):
    """
    Retrieve a node by ID from the frame.

    Raises:
        ValueError: If node_id is not found.
    """
    for node in frame.nodes:
        if node.id == node_id:
            return node
    raise ValueError(f"Node {node_id} not found in frame.")