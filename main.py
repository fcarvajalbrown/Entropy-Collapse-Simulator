"""
main.py
=======
Command-line entry point for the planar entropy-based structural analysis tool.

Three analysis modes are available via --mode:

    simulate   : progressive-collapse simulation with entropy-based detection
    robustness : Entropy Robustness Index R_S by notional member removal (ALP)
    criteria   : head-to-head comparison of four collapse criteria

Usage:
    python main.py --list                                  # list scenarios
    python main.py --mode simulate   --scenario pratt_bridge --save
    python main.py --mode robustness --scenario building_2d
    python main.py --mode criteria   --scenario pratt_bridge --load-step 0.15

Common arguments:
    --mode      : simulate (default), robustness, or criteria
    --scenario  : frame name from the registry (default: 2d_simple)
    --method    : collapse detection for simulate mode (zscore or threshold)
    --steps     : maximum steps for simulate/criteria modes
    --save      : save figures to disk instead of displaying them
    --list      : print available scenarios and exit
"""

import argparse
import os

from structure.frames import (
    frame_2d_simple, frame_building_2d, frame_pratt_bridge, frame_vogel_six_storey
)
from simulation.runner import run
from entropy import robustness as robustness_mod
from analysis import criteria as criteria_mod
from visualization.graph_view import plot_frame, plot_collapse_sequence
from visualization.entropy_plot import plot_entropy
from visualization.animation import animate_collapse


FRAME_BUILDERS = {
    "2d_simple":      frame_2d_simple.build,
    "building_2d":    frame_building_2d.build,
    "building_large": lambda: frame_building_2d.build(n_bays=3, n_stories=6),
    "pratt_bridge":   frame_pratt_bridge.build,
    "vogel_six_storey": frame_vogel_six_storey.build,
}


def main():
    """Parse CLI arguments and dispatch to the selected analysis mode."""
    args = _parse_args()

    if args.list:
        print("Available scenarios:")
        for name in FRAME_BUILDERS:
            print(f"  {name}")
        return

    if args.scenario not in FRAME_BUILDERS:
        print(f"Unknown scenario '{args.scenario}'. Use --list to see available options.")
        return

    frame = FRAME_BUILDERS[args.scenario]()

    if args.mode == "robustness":
        _run_robustness(args, frame)
        return
    if args.mode == "criteria":
        _run_criteria(args, frame)
        return
    _run_simulate(args, frame)


def _run_robustness(args, frame):
    """Compute and print the Entropy Robustness Index for the frame."""
    report = robustness_mod.analyze(frame)
    print(f"Entropy Robustness Index : {report.frame_name}")
    print(f"  Intact entropy H0          : {report.intact_entropy_norm:.4f}")
    print(f"  Robustness index R_S       : {report.robustness_index:.4f}  (0 fragile -> 1 robust)")
    print(f"  R_S (stable removals only) : {report.robustness_index_stable_only:.4f}")
    print(f"  Mechanism fraction         : {report.mechanism_fraction:.2f}")
    print(f"  Worst-case H_k             : {report.worst_case_entropy_norm:.4f}")
    print(f"  Critical member            : {report.critical_member}")
    if report.unstable_members:
        print(f"  Removal causes mechanism   : members {report.unstable_members}")
    print("  Criticality ranking (member: entropy drop):")
    for mid, drop in report.ranking()[:10]:
        print(f"    member {mid:>3} : {drop:+.4f}")


def _run_criteria(args, frame):
    """Run and print the four-criteria comparison for the frame."""
    cmp = criteria_mod.compare(
        frame,
        max_steps=args.steps,
        load_factor_start=1.0,
        load_factor_step=args.load_step,
    )
    print(f"Collapse-criteria comparison : {cmp.frame_name}")
    print(f"  Steps run        : {len(cmp.history)}")
    print(f"  Failure sequence : {cmp.failed_sequence}")
    print("  First-trigger step (and load factor) per criterion:")
    for name in ("displacement", "dcr", "energy", "entropy"):
        step = cmp.first_trigger[name]
        if step is None:
            print(f"    {name:<13}: not triggered")
        else:
            lf = cmp.history[step].load_factor
            print(f"    {name:<13}: step {step:>3}  (load factor {lf:.2f})")


def _run_simulate(args, frame):
    """
    Run the progressive-collapse simulation and display results:
      1. Frame view at the final step with energy heatmap
      2. Collapse sequence overlay (if collapse was detected)
      3. Entropy evolution plot (S, dS/dt, Gini index)
      4. Animation (if --animate is set)
    """
    print(f"Running scenario : {args.scenario}")
    print(f"Detection method : {args.method}")
    print(f"Max steps        : {args.steps}")
    print(f"Load factor step : {args.load_step}")
    print()

    result = run(
        frame,
        max_steps=args.steps,
        collapse_method=args.method,
        load_factor_start=1.0,
        load_factor_step=args.load_step,
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

    if args.animate:
        output_path = os.path.join(
            save_dir if args.save else ".",
            f"collapse_{args.scenario}.{args.animate_fmt}"
        )
        animate_collapse(
            result=result,
            frame=frame,
            output_path=output_path,
            fps=args.fps,
        )


def _parse_args() -> argparse.Namespace:
    """
    Define and parse CLI arguments.

    Returns:
        Parsed argparse.Namespace object.
    """
    parser = argparse.ArgumentParser(
        description="Planar entropy-based structural analysis tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--mode", type=str, default="simulate",
        choices=["simulate", "robustness", "criteria"],
        help="Analysis mode (default: simulate)"
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
        "--load-step", dest="load_step", type=float, default=0.2,
        help="Load factor increment per step (default: 0.2). "
             "Set to 0.0 for static loading at design load."
    )
    parser.add_argument(
        "--save", action="store_true",
        help="Save figures to output_figures/ instead of displaying"
    )
    parser.add_argument(
        "--animate", action="store_true",
        help="Produce an animation of energy redistribution (saved to file)"
    )
    parser.add_argument(
        "--animate-fmt", dest="animate_fmt", type=str, default="gif",
        choices=["gif", "mp4"],
        help="Animation output format (default: gif)"
    )
    parser.add_argument(
        "--fps", type=int, default=10,
        help="Frames per second for animation (default: 10)"
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List available scenarios and exit"
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()