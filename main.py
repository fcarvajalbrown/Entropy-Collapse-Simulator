"""
main.py
=======
Entry point for the Entropy-Based Progressive Collapse Simulator.

Usage:
    python main.py                          # runs default scenario (2d_simple)
    python main.py --scenario 3d_redundant
    python main.py --scenario 2d_simple --method threshold --steps 200
    python main.py --list                   # list available scenarios

Arguments:
    --scenario  : Scenario name from the registry (default: 2d_simple)
    --method    : Collapse detection method — zscore or threshold (default: zscore)
    --steps     : Maximum simulation steps (default: 100)
    --save      : Save figures to disk instead of displaying them
    --list      : Print available scenarios and exit
"""

import argparse
import os

from structure.frames import frame_2d_simple, frame_3d_redundant
from simulation.runner import run
from visualization.graph_view import plot_frame, plot_collapse_sequence
from visualization.entropy_plot import plot_entropy


FRAME_MODULES = {
    "2d_simple":     frame_2d_simple,
    "3d_redundant":  frame_3d_redundant,
}


def main():
    """
    Parse CLI arguments, run the selected scenario, and display results.

    Loads the frame, runs the simulation, then shows:
      1. Frame view at the final step with energy heatmap
      2. Collapse sequence overlay (if collapse was detected)
      3. Entropy evolution plot (S, dS/dt, Gini index)
    """
    args = _parse_args()

    if args.list:
        print("Available scenarios:")
        for name in FRAME_MODULES:
            print(f"  {name}")
        return

    if args.scenario not in FRAME_MODULES:
        print(f"Unknown scenario '{args.scenario}'. Use --list to see available options.")
        return

    print(f"Running scenario : {args.scenario}")
    print(f"Detection method : {args.method}")
    print(f"Max steps        : {args.steps}")
    print()

    frame = FRAME_MODULES[args.scenario].build()

    result = run(
        frame,
        max_steps=args.steps,
        collapse_method=args.method
    )

    # --- Report summary ---
    print(f"Simulation complete: {result.frame_name}")
    print(f"  Steps run        : {len(result.energy_history)}")
    print(f"  Collapse detected: {result.collapse_detected}")
    if result.collapse_detected:
        print(f"  Collapse at step : {result.collapse_step}")
    if result.failed_sequence:
        print(f"  Failure sequence : {result.failed_sequence}")
    print()

    # --- Visualization ---
    save_dir = "output_figures" if args.save else None
    if args.save:
        os.makedirs(save_dir, exist_ok=True)
        print(f"Saving figures to: {save_dir}/")

    final_energy = result.energy_history[-1]
    final_entropy = result.entropy_history[-1]

    plot_frame(
        frame=frame,
        energy_state=final_energy,
        entropy_record=final_entropy,
        step=final_entropy.step,
        show=not args.save,
        save_path=os.path.join(save_dir, "frame_final.png") if args.save else None
    )

    if result.collapse_detected:
        plot_collapse_sequence(
            frame=frame,
            failed_sequence=result.failed_sequence,
            show=not args.save,
            save_path=os.path.join(save_dir, "collapse_sequence.png") if args.save else None
        )

    plot_entropy(
        result=result,
        show=not args.save,
        save_path=os.path.join(save_dir, "entropy_analysis.png") if args.save else None
    )


def _parse_args() -> argparse.Namespace:
    """
    Define and parse CLI arguments.

    Returns:
        Parsed argparse.Namespace object.
    """
    parser = argparse.ArgumentParser(
        description="Entropy-Based Progressive Collapse Simulator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--scenario", type=str, default="2d_simple",
        help="Scenario name to run (default: 2d_simple)"
    )
    parser.add_argument(
        "--method", type=str, default="zscore", choices=["zscore", "threshold"],
        help="Collapse detection method (default: zscore)"
    )
    parser.add_argument(
        "--steps", type=int, default=100,
        help="Maximum simulation steps (default: 100)"
    )
    parser.add_argument(
        "--save", action="store_true",
        help="Save figures to output_figures/ instead of displaying"
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List available scenarios and exit"
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()