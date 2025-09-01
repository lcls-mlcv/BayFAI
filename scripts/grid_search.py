import argparse
import json
import numpy as np

from bayfai.setup import generate_powder, build_detector, define_calibrant, min_intensity
from bayfai.optimization import BayesGeomOpt
from bayfai.plots import plot_pairwise_heatmaps

def main(args):
    # Generate powder
    powder = generate_powder(args.powder_path, args.detname, smooth=args.smooth)

    # Build detector
    stacked_shape = (powder.shape[0] * powder.shape[1], powder.shape[2])
    detector = build_detector(args.in_file, stacked_shape)

    # Define calibrant
    calibrant = define_calibrant(args.calibrant, args.wavelength)

    # Compute minimum intensity threshold
    Imin = min_intensity(powder, args.threshold)

    # Initialize Optimizer
    optimizer = BayesGeomOpt(
        exp=args.exp,
        run=args.run,
        detector=detector,
        powder=powder,
        calibrant=calibrant,
        fixed=args.fixed,
    )

    # Create Search Space
    X, _ = optimizer.create_search_space(
        center=args.center,
        bounds=args.bounds,
        resolution=args.resolution
    )

    # Grid Search
    score_map = np.zeros(X.shape[0])
    best_score = -np.inf
    for i in range(X.shape[0]):
        sample = X[i]
        score = optimizer.score(sample, Imin, args.max_rings, args.rtol)
        score_map[i] = score
        if i == 0 or score > best_score:
            best_score = score
            best_params = sample

    print(f"Best distance: {best_params[0]}")
    print(f"Best shift-x: {best_params[1]}")
    print(f"Best shift-y: {best_params[2]}")
    print(f"Best tilt-x: {best_params[3]}")
    print(f"Best tilt-y: {best_params[4]}")
    print(f"Best tilt-z: {best_params[5]}")
    print(f"Best score: {best_score}")

    # Save results
    np.savez_compressed(
        f"{args.out_dir}/grid_search_score_{args.exp}_r{args.run:04d}.npz",
        X=X,
        scores=score_map
    )

    # Plot results
    fig = plot_pairwise_heatmaps(X, score_map, optimizer.order, best_params)
    fig.savefig(f"{args.out_dir}/grid_search_heatmap_{args.exp}_r{args.run:04d}_rtol_{str(args.rtol).replace('.', '')}.png", dpi=300)

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