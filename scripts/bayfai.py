"""Run a single BayFAI geometry optimization from the command line.

BayFAI parallelizes the detector-distance scan across MPI ranks, so this script
must be launched under an MPI runner, e.g.::

    source /sdf/group/lcls/ds/ana/sw/conda2/manage/bin/psconda.sh
    export PYTHONPATH=/sdf/group/lcls/ds/tools/LCLSGeom/src

    mpirun -n 20 python scripts/bayfai.py \
        --exp mfx101211025 --run 23 --detname jungfrau --calibrant LaB6 \
        --h5 /path/to/powder.h5 --out_file ./geom/23-end.data

The number of ranks sets how many candidate distances are scanned, each centred
on ``center["dist"]`` and spaced by ``resolutions["dist"]``.

For the full benchmark suite use ``scripts/run_benchmark.py`` instead, which
submits these optimizations through LUTE.
"""

import argparse
import json
import logging
import os
from time import time

from bayfai.optimization import BayFAIOpt

DEFAULT_CENTER = {
    "dist": 0.1,
    "poni1": 0.0,
    "poni2": 0.0,
    "rot1": 0.0,
    "rot2": 0.0,
    "rot3": 0.0,
}

DEFAULT_BOUNDS = {
    "dist": (-0.05, 0.05),
    "poni1": (-0.005, 0.005),
    "poni2": (-0.005, 0.005),
    "rot1": (-0.1, 0.1),
    "rot2": (-0.1, 0.1),
    "rot3": (-0.1, 0.1),
}

DEFAULT_RESOLUTIONS = {
    "dist": 0.001,
    "poni1": 0.0001,
    "poni2": 0.0001,
    "rot1": 0.02,
    "rot2": 0.02,
    "rot3": 0.02,
}


def main(args):
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    optimizer = BayFAIOpt(exp=args.exp, run=args.run)

    optimizer.setup(
        detname=args.detname,
        h5=args.h5,
        Imin=args.Imin,
        calibrant=args.calibrant,
        fixed=args.fixed,
        wavelength=args.wavelength,
    )

    tic = time()
    optimizer.bayfai_opt(
        center=args.center,
        bounds=args.bounds,
        res=args.resolutions,
        n_samples=args.n_samples,
        n_iterations=args.n_iterations,
        Imin=optimizer.Imin,
        max_rings=args.max_rings,
        pts_per_deg=args.pts_per_deg,
        beta=args.beta,
        prior=args.prior,
        step=args.step,
        lbda=args.lbda,
        seed=args.seed,
    )
    toc = time()

    if optimizer.rank != 0:
        return

    params = optimizer.params
    print(f"Optimization took {toc - tic:.2f} seconds")
    print(f"Detector distance (m): {params[0]:.6f}")
    print(f"poni1 (m): {params[1]:.6f}")
    print(f"poni2 (m): {params[2]:.6f}")
    print(f"rot1 (rad): {params[3]:.2e}")
    print(f"rot2 (rad): {params[4]:.2e}")
    print(f"rot3 (rad): {params[5]:.2e}")
    print(f"Best score: {optimizer.neglog_score:.2e}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out_file)), exist_ok=True)
    optimizer.update_geometry(args.out_file)
    print(f"Wrote optimized geometry to {args.out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="BayFAI Geometry Optimization (launch under mpirun/srun)"
    )

    # --- Experiment ---
    parser.add_argument("--exp", type=str, required=True, help="Experiment name")
    parser.add_argument("--run", type=int, required=True, help="Run number")
    parser.add_argument(
        "--detname",
        type=str,
        required=True,
        help="Detector name (jungfrau, jungfrau4M, epix10k2M, Epix10kaQuad0, Rayonix...)",
    )

    # --- I/O ---
    parser.add_argument(
        "--h5",
        type=str,
        required=True,
        help="Path to the smalldata .h5 (or a .npy) powder file",
    )
    parser.add_argument(
        "--out_file",
        type=str,
        required=True,
        help="Output psana .data geometry file. Sibling .poni and .geom files are written next to it.",
    )

    # --- Calibration ---
    parser.add_argument(
        "--calibrant",
        type=str,
        required=True,
        help="pyFAI calibrant name (AgBh, LaB6, CeO2...)",
    )
    parser.add_argument(
        "--wavelength",
        type=float,
        default=1e-10,
        help="Wavelength in m. If left at the default, it is read from the h5 file.",
    )

    # --- Search space ---
    parser.add_argument(
        "--fixed",
        type=str,
        default='["rot3"]',
        help='JSON list of parameters to keep fixed, e.g. \'["rot1","rot2","rot3"]\'',
    )
    parser.add_argument(
        "--center", type=str, default=None, help="JSON dict: center of the search space"
    )
    parser.add_argument(
        "--bounds",
        type=str,
        default=None,
        help="JSON dict: per-parameter bounds relative to the center",
    )
    parser.add_argument(
        "--resolutions",
        type=str,
        default=None,
        help="JSON dict: per-parameter grid resolutions",
    )

    # --- Hyperparameters ---
    parser.add_argument("--n_samples", type=int, default=20, help="Initial GP samples")
    parser.add_argument(
        "--n_iterations", type=int, default=80, help="Bayesian optimization iterations"
    )
    parser.add_argument(
        "--max_rings", type=int, default=6, help="Maximum number of rings to score"
    )
    parser.add_argument(
        "--pts_per_deg",
        type=float,
        default=0.5,
        help="Control points extracted per azimuthal degree",
    )
    parser.add_argument(
        "--Imin",
        type=float,
        default=95,
        help="Intensity percentile threshold for Bragg peak detection",
    )
    parser.add_argument(
        "--beta", type=float, default=1.96, help="UCB exploration-exploitation trade-off"
    )
    parser.add_argument(
        "--step", type=int, default=5, help="Refinement box half-width, in grid steps"
    )
    parser.add_argument(
        "--lbda", type=float, default=0.1, help="Uncertainty penalty in winner selection"
    )
    parser.add_argument(
        "--no-prior",
        dest="prior",
        action="store_false",
        default=True,
        help="Draw initial samples uniformly at random instead of from a Gaussian prior around the center",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed for reproducibility"
    )

    args = parser.parse_args()
    args.fixed = json.loads(args.fixed)
    args.center = json.loads(args.center) if args.center else dict(DEFAULT_CENTER)
    args.bounds = json.loads(args.bounds) if args.bounds else dict(DEFAULT_BOUNDS)
    args.resolutions = (
        json.loads(args.resolutions)
        if args.resolutions
        else dict(DEFAULT_RESOLUTIONS)
    )
    main(args)
