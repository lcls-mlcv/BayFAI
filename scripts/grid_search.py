import argparse
import json
from time import time

from bayfai.optimization import BayFAIOpt

def main(args):
    # Initialize Optimizer
    optimizer = BayFAIOpt(
        exp=args.exp,
        run=args.run,
    )

    # Setup Optimization
    optimizer.setup(
        detname=args.detname,
        powder=args.powder,
        smooth=args.smooth,
        calibrant=args.calibrant,
        wavelength=args.wavelength,
        fixed=args.fixed,
        in_file=args.in_file,
    )

    # Run Grid Search
    tic = time()
    optimizer.grid_search(
        center=args.center,
        bounds=args.bounds,
        res=args.resolutions,
        max_rings=args.max_rings,
        score=args.score,
        out_dir=args.out_dir,
    )
    toc = time()
    print(f"Optimization took {toc - tic:.2f} seconds")

    print(f"Best History: {optimizer.best_dist}", flush=True)
    print(f"Best Distance: {optimizer.scan[optimizer.best_dist]["params"][0]}", flush=True)
    print(f"Best Score: {optimizer.best_score}", flush=True)
    print("Best Geometry", flush=True)
    print(f"Best Distance: {optimizer.scan[best_dist]["params"][best_index][0]}")
    print(f"Best X-shift: {optimizer.scan[best_dist]["params"][best_index][1]}")
    print(f"Best Y-shift: {optimizer.scan[best_dist]["params"][best_index][2]}")
    print(f"Best Rot-x: {optimizer.scan[best_dist]["params"][best_index][3]}")
    print(f"Best Rot-y: {optimizer.scan[best_dist]["params"][best_index][4]}")
    print(f"Best Rot-z: {optimizer.scan[best_dist]["params"][best_index][5]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BayFAI Geometry Optimization")

    # --- Experiment Arguments ---
    parser.add_argument("--exp", type=str, required=True, help="Experiment name")
    parser.add_argument("--run", type=int, required=True, help="Run number")
    parser.add_argument("--detname", type=str, required=True, help="Detector name")

    # --- I/O Arguments ---
    parser.add_argument("--in_file", type=str, required=True, help="Path to the input metrology file")
    parser.add_argument("--out_dir", type=str, required=True, help="Path to the output directory")

    # --- Powder Arguments ---
    parser.add_argument("--powder_path", type=str, required=True, help="Path to the powder data file")
    parser.add_argument("--smooth", action="store_true", help="Apply smoothing to the powder image")
    parser.add_argument("--threshold", type=float, default=95, help="Percentile to use for thresholding the powder image")

    # --- Calibration Arguments ---
    parser.add_argument("--calibrant", type=str, required=True, help="Name of the calibrant")
    parser.add_argument("--wavelength", type=float, required=True, help="Wavelength of the X-ray source")

    # --- Search Space Arguments ---
    parser.add_argument("--fixed", type=str, default=["rot3"], help="List of parameters to keep fixed during optimization")
    parser.add_argument("--center", type=str, required=True, help="Center of the search space")
    parser.add_argument("--bounds", type=str, required=True, help="Per-parameter size of the search space around the center")
    parser.add_argument("--resolution", type=str, required=True, help="Per-parameter resolution of the search space")

    # --- BayFAI Hyperparameters ---
    parser.add_argument("--n_init", type=int, default=100, help="Number of initial samples")
    parser.add_argument("--n_iter", type=int, default=400, help="Number of bayesian optimization iterations")
    parser.add_argument("--max_rings", type=int, default=6, help="Maximum number of rings to use for scoring")
    parser.add_argument("--rtol", type=float, default=1e-2, help="Relative tolerance in q-space for masking ring pixels")
    parser.add_argument("--beta", type=float, default=1.96, help="Exploration-exploitation trade-off parameter for UCB")

    # --- Random Arguments ---
    parser.add_argument("--prior", action="store_true", help="If true, sample points around the center, else sample at random")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    args = parser.parse_args()
    args.fixed = json.loads(args.fixed)
    args.center = json.loads(args.center)
    args.bounds = json.loads(args.bounds)
    args.resolution = json.loads(args.resolution)
    main(args)