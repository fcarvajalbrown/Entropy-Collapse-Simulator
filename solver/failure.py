"""
solver/failure.py
=================
Evaluates member failure criteria and marks failed members in the frame.
A failed member is excluded from stiffness assembly in subsequent steps.

Current criterion: axial force exceeds member capacity (F > F_yield).
Bending failure (M > M_plastic) can be added later without breaking the interface.

Consumed by simulation/runner.py after each equilibrium solve.
"""

from core.models import FrameData, EnergyState, MemberState


def check_and_apply_failures(frame: FrameData, energy_state: EnergyState) -> list[int]:
    """
    Check all members for failure and mark them in the frame.

    Iterates over member states, compares axial force against capacity,
    and sets member.failed = True for any that exceed the limit.

    Args:
        frame: Frame definition (modified in-place — failed flags updated).
        energy_state: Current energy/force state from the equilibrium solver.

    Returns:
        List of member IDs that newly failed this step.
    """
    newly_failed = []

    for ms in energy_state.member_states:
        if ms.failed:
            continue

        member = _get_member(frame, ms.member_id)
        capacity = _axial_capacity(member)

        if abs(ms.axial_force) >= capacity:
            member.failed = True
            newly_failed.append(member.id)

    return newly_failed


def _axial_capacity(member) -> float:
    """
    Compute the axial load capacity of a member.

    Uses a yield-stress-based approach: F_yield = sigma_y * A.
    Assumes steel with sigma_y = 250 MPa by default if not set on the member.

    TODO: Add member.sigma_y field to Member model for custom materials.

    Args:
        member: Member with cross-sectional area A.

    Returns:
        Axial capacity in Newtons.
    """
    sigma_y = getattr(member, "sigma_y", 250e6)  # Default: 250 MPa steel
    return sigma_y * member.A


def _get_member(frame: FrameData, member_id: int):
    """
    Retrieve a member by ID from the frame.

    Args:
        frame: Frame definition.
        member_id: ID to look up.

    Returns:
        Member object.

    Raises:
        ValueError: If member_id is not found.
    """
    for member in frame.members:
        if member.id == member_id:
            return member
    raise ValueError(f"Member {member_id} not found in frame.")


def all_failed(frame: FrameData) -> bool:
    """
    Check if all members in the frame have failed (total collapse).

    Args:
        frame: Current frame state.

    Returns:
        True if every member is marked as failed.
    """
    return all(m.failed for m in frame.members)