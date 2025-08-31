import argparse

from bayfai.setup import generate_powder, build_detector, define_calibrant, min_intensity
from bayfai.optimization import BayesGeomOpt

def main(args):
    # Generate powder
    powder, raw_powder = generate_powder(args.powder_path, args.detname, smooth=args.smooth)

    # Build detector
    detector_shape = (powder.shape[0] * powder.shape[1], powder.shape[2])
    detector = build_detector(args.in_file, detector_shape)

    # Define calibrant
    calibrant = define_calibrant(args.calibrant, args.wavelength)

    # Compute minimum intensity threshold
    Imin = min_intensity(raw_powder, args.threshold)

    # Initialize Optimizer
    optimizer = BayesGeomOpt(
        exp=args.exp,
        run=args.run,
        detector=detector,
        powder=powder,
        calibrant=calibrant,
    )

    # Run Bayesian Optimization
    optim_params = {
        "center": args.center,
        "bounds": args.bounds,
        "res": args.resolution,
        "n_samples": args.n_init,
        "n_iterations": args.n_iter,
        "Imin": Imin,
        "max_rings": args.max_rings,
        "rtol": args.rtol,
        "prior": args.prior,
        "seed": args.seed,
    }
    result = optimizer.bayes_opt(**optim_params)

    print(f"Best distance: {result['params'][0]}")
    print(f"Best shift-x: {result['params'][1]}")
    print(f"Best shift-y: {result['params'][2]}")
    print(f"Best tilt-x: {result['params'][3]}")
    print(f"Best tilt-y: {result['params'][4]}")
    print(f"Best tilt-z: {result['params'][5]}")
    print(f"Best score: {result['score']}")
    print(f"Residual: {result['residual']}")

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
    parser.add_argument("--center", type=dict, required=True, help="Center of the search space")
    parser.add_argument("--bounds", type=dict, required=True, help="Per-parameter size of the search space around the center")
    parser.add_argument("--resolution", type=dict, required=True, help="Per-parameter resolution of the search space")

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
    main(args)