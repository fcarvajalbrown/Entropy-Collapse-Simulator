"""
solver/failure.py
=================
Evaluates member failure criteria and marks failed members in the frame.

Failure criterion: total internal force magnitude exceeds member capacity.
    F_magnitude >= sigma_y * A

Since MemberState.axial_force now stores the full internal force magnitude
(from equilibrium.py), this correctly catches both axial and bending-dominated
failures without needing separate shear or bending checks.

Consumed by simulation/runner.py after each equilibrium solve.
"""

from core.models import FrameData, EnergyState


def check_and_apply_failures(frame: FrameData, energy_state: EnergyState) -> list[int]:
    """
    Check all members for failure and mark them in the frame.

    Compares internal force magnitude against yield capacity (sigma_y * A).
    Sets member.failed = True for any that exceed the limit.

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
        capacity = _capacity(member)

        if ms.axial_force >= capacity:
            member.failed = True
            newly_failed.append(member.id)

    return newly_failed


def _capacity(member) -> float:
    """
    Compute the load capacity of a member as sigma_y * A.

    Uses member.material.sigma_y — always explicitly defined in frame files.

    Args:
        member: Member with cross-sectional area A.

    Returns:
        Capacity in Newtons.
    """
    return member.sigma_y * member.A


def _get_member(frame: FrameData, member_id: int):
    """
    Retrieve a member by ID from the frame.

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