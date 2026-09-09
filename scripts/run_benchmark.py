"""Submit the BayFAI benchmark suite for a given hutch through LUTE.

Reads the reference configs shipped in ``benchmark/yamls/<hutch>/``, applies any
hyperparameter overrides given on the command line, writes the resulting configs
into the timestamped results folder, and submits one SLURM job per config.

The configs tracked in git are never modified: everything generated lands under
``<work_dir>/results/test_<hutch>_<timestamp>/``.

Usage
-----
    python scripts/run_benchmark.py --hutch cxi
    python scripts/run_benchmark.py --hutch mec --n_iterations 120 --dry-run
"""

import argparse
import subprocess
from datetime import datetime
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LUTE = Path(
    "/sdf/data/lcls/ds/prj/prjlute22/results/benchmarks/geom_opt/lute"
)
HUTCHES = ["cxi", "mec", "mfx_psana1", "mfx_psana2"]

# Hyperparameters that live under the BayFAI.bayfai_params block. Only the ones
# actually supplied on the command line are written, so the values committed in
# the reference configs stay in force otherwise.
HYPERPARAMETERS = (
    "n_samples",
    "n_iterations",
    "max_rings",
    "pts_per_deg",
    "Imin",
    "beta",
    "step",
    "lbda",
)


def submit_task(
    executable,
    yaml_file,
    task,
    ncores,
    partition,
    account,
    exp=None,
    run=None,
    psana2=False,
):
    """Construct the SLURM command to submit the job."""
    command = f"{executable}"
    command += f" -t {task}"
    command += f" -c {yaml_file}"
    if exp is not None:
        command += f" -e {exp}"
        command += f" -r {run}"
    if psana2:
        command += " --psana2"
    command += f" --ntasks={ncores}"
    command += " --nodes=1"
    command += f" --partition={partition}"
    command += f" --account={account}"
    return command


def build_config(config_path, args, test_folder):
    """Load a reference config and return (config docs, exp, run, detname)."""
    with open(config_path, "r") as f:
        docs = list(yaml.safe_load_all(f))

    exp = run = detname = None
    for doc in docs:
        if not doc:
            continue
        if "experiment" in doc:
            exp = doc["experiment"]
        if "run" in doc:
            run = int(doc["run"])
        if "work_dir" in doc:
            doc["work_dir"] = str(test_folder)

    for doc in docs:
        if not doc or "BayFAI" not in doc:
            continue
        bayfai = doc["BayFAI"]
        detname = bayfai["detname"]

        # `h5` is intentionally left as committed in the reference config. It points at
        # the smalldata HDF5 powder, which carries the per-run photon energy; swapping in
        # the assembled .npy copy would silently fall back to a default 1 A wavelength.

        # MEC runs four ePix10ka quads per run, so the per-run default that LUTE
        # would derive ({work_dir}/geom/{run}-end.data) would have all four
        # overwrite each other. Key the output on the detector instead.
        geom_folder = test_folder / "geom" / f"{exp}"
        geom_folder.mkdir(parents=True, exist_ok=True)
        if detname.lower().startswith("epix10kaquad"):
            quad = detname[-1]
            bayfai["out_file"] = str(geom_folder / f"{run}-end_Q{quad}.data")
        else:
            bayfai["out_file"] = str(geom_folder / f"{run}-end.data")

        params = bayfai.setdefault("bayfai_params", {})
        for key in HYPERPARAMETERS:
            value = getattr(args, key)
            if value is not None:
                params[key] = value
        if args.prior is not None:
            params["prior"] = args.prior
        if args.seed is not None:
            params["seed"] = None if args.seed == 0 else args.seed

    if exp is None or run is None:
        raise ValueError(f"{config_path} is missing 'experiment' and/or 'run'")
    return docs, exp, run, detname


def main(args):
    yaml_folder = REPO_ROOT / "benchmark" / "yamls" / args.hutch
    if not yaml_folder.is_dir():
        raise SystemExit(f"No benchmark configs for hutch '{args.hutch}' in {yaml_folder}")

    datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    test_folder = Path.cwd() / "results" / f"test_{args.hutch}_{datetime_str}"
    config_folder = test_folder / "configs"
    config_folder.mkdir(parents=True, exist_ok=True)
    (test_folder / "figs").mkdir(parents=True, exist_ok=True)

    if args.hutch == "mfx_psana2":
        task, psana2 = "BayFAIOptimizer2", True
    else:
        task, psana2 = "BayFAIOptimizer", False
    executable = Path(args.lute) / "launch_scripts" / "submit_slurm.sh"

    configs = sorted(yaml_folder.glob("*.yaml"))
    if not configs:
        raise SystemExit(f"No .yaml files found in {yaml_folder}")
    print(f"Submitting {len(configs)} {args.hutch} benchmark configs -> {test_folder}")

    for config_path in configs:
        docs, exp, run, _ = build_config(config_path, args, test_folder)

        out_config = config_folder / config_path.name
        with open(out_config, "w") as f:
            yaml.safe_dump_all(docs, f, default_flow_style=False, sort_keys=True)

        cmd = submit_task(
            executable,
            out_config,
            task,
            args.ntasks,
            args.partition,
            args.account,
            exp,
            run,
            psana2,
        )

        if args.dry_run:
            print(cmd)
            continue

        try:
            result = subprocess.run(
                cmd,
                shell=True,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            print(f"Submitted {config_path.name}: {result.stdout.strip()}")
            if result.stderr.strip():
                print("Standard Error:", result.stderr.strip())
        except subprocess.CalledProcessError as e:
            print(f"Error submitting {config_path.name}: return code {e.returncode}")
            print("Error Output:", e.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the BayFAI benchmark suite")

    parser.add_argument(
        "--hutch",
        type=str,
        required=True,
        choices=HUTCHES,
        help="Which hutch benchmark to run",
    )

    # BayFAI hyperparameters. Default None means "leave the config's value alone".
    parser.add_argument("--n_samples", type=int, default=None, help="Initial GP samples")
    parser.add_argument(
        "--n_iterations", type=int, default=None, help="Bayesian optimization iterations"
    )
    parser.add_argument(
        "--max_rings", type=int, default=None, help="Maximum number of rings to score"
    )
    parser.add_argument(
        "--pts_per_deg",
        type=float,
        default=None,
        help="Control points extracted per azimuthal degree",
    )
    parser.add_argument(
        "--Imin",
        type=float,
        default=None,
        help="Intensity percentile threshold for Bragg peak detection",
    )
    parser.add_argument(
        "--beta", type=float, default=None, help="UCB exploration-exploitation trade-off"
    )
    parser.add_argument(
        "--step", type=int, default=None, help="Refinement box half-width, in grid steps"
    )
    parser.add_argument(
        "--lbda",
        type=float,
        default=None,
        help="Uncertainty penalty weight in winner selection",
    )
    parser.add_argument(
        "--no-prior",
        dest="prior",
        action="store_false",
        default=None,
        help="Draw initial samples uniformly at random instead of from a Gaussian prior",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (0 means no seed)",
    )

    # Submission settings.
    parser.add_argument(
        "--lute", type=str, default=str(DEFAULT_LUTE), help="Path to the LUTE clone to use"
    )
    parser.add_argument("--partition", type=str, default="milano", help="SLURM partition")
    parser.add_argument(
        "--account", type=str, default="lcls:prjlute22", help="SLURM account"
    )
    parser.add_argument(
        "--ntasks", type=int, default=120, help="MPI ranks per job (distances scanned)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write the configs and print the submission commands without submitting",
    )

    main(parser.parse_args())
