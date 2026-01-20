import os
import subprocess
from pathlib import Path
from datetime import datetime
import yaml
import argparse

def submit_task(executable, yaml_file, task, ncores, partition, account, exp=None, run=None, psana2=False):
    """
    Construct the SLURM command to submit the job
    """
    command = f'{executable}'
    command += f' -t {task}'
    command += f' -c {yaml_file}'
    if exp is not None:
        command += f' -e {exp}'
        command += f' -r {run}'
    if psana2:
        command += f' --psana2'
    command += f' --ntasks={ncores}'
    command += ' --nodes=1'
    command += f' --partition={partition}'
    command += f' --account={account}'
    return command

def main(args):
    
    # Parse arguments
    hutch = args.hutch
    n_samples = args.n_samples
    n_iterations = args.n_iterations
    max_rings = args.max_rings
    prior = args.prior
    beta = args.beta
    step = args.step
    seed = args.seed
    smooth = args.smooth

    # Set up directories
    work_dir = Path.cwd()
    yaml_folder = Path(f"../BayFAI/benchmark/yamls/{hutch}")
    powder_folder = Path(f"../BayFAI/benchmark/powder")
    results_folder = work_dir / "results"
    datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    test_folder_name = f"test_{hutch}_{datetime_str}"
    test_folder = results_folder / test_folder_name
    test_folder.mkdir(parents=True, exist_ok=True)

    # Modify YAML files
    yamls = yaml_folder.glob("*.yaml")
    for config_path in yamls:
        with open(config_path, "r") as f:
            config = list(yaml.safe_load_all(f))
        for doc in config:
            keys = doc.keys()
            if 'work_dir' in keys:
                doc['work_dir'] = str(test_folder)
            if 'experiment' in keys:
                exp = doc['experiment']
            if 'run' in keys:
                run = doc['run']
            geom_folder = test_folder / "geom" / f"{exp}"
            geom_folder.mkdir(parents=True, exist_ok=True)
            fig_folder = test_folder / "figs"
            fig_folder.mkdir(parents=True, exist_ok=True)
            if 'BayFAI' in keys:
                doc['BayFAI'].setdefault('bo_params', {})
                if hutch != 'mfx_psana2':
                    in_file = doc['BayFAI']['in_file']
                if hutch == 'mec':
                    quad = os.path.basename(in_file).split('.')[0][-1]
                    doc['BayFAI']['out_file'] = f"{geom_folder}/{run}-end_Q{quad}.data"
                else:
                    doc['BayFAI']['out_file'] = f"{geom_folder}/{run}-end.data"
                doc['BayFAI']['powder'] = f"{powder_folder}/{exp}_Run{run:0>4}.npy"
                doc['BayFAI']['preprocess'] = smooth
                doc['BayFAI']['bo_params']['n_samples'] = n_samples
                doc['BayFAI']['bo_params']['n_iterations'] = n_iterations
                doc['BayFAI']['bo_params']['max_rings'] = max_rings
                doc['BayFAI']['bo_params']['prior'] = prior
                doc['BayFAI']['bo_params']['beta'] = beta
                doc['BayFAI']['bo_params']['step'] = step
                if seed==0:
                    doc['BayFAI']['bo_params']['seed'] = None
                else:
                    doc['BayFAI']['bo_params']['seed'] = seed

        with open(config_path, "w") as f:
            yaml.safe_dump_all(config, f)

        # Submit SLURM jobs
        executable = "/sdf/data/lcls/ds/prj/prjlute22/results/benchmarks/geom_opt/lute/launch_scripts/submit_slurm.sh"
        if hutch=='mfx_psana2':
            task = "BayFAIOptimizer2"
            psana2 = True
        else:
            task = "BayFAIOptimizer"
            psana2 = False
        ncores = 120
        partition = "milano"
        account = "lcls:prjlute22"

        cmd = submit_task(executable, config_path, task, ncores, partition, account, exp, run, psana2)

        try:
            result = subprocess.run(
                cmd,
                shell=True,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            print("SLURM job submitted successfully!")
            print("Standard Output:", result.stdout)
            print("Standard Error:", result.stderr)

        except subprocess.CalledProcessError as e:
            # Handle errors in the subprocess call
            print(f"Error occurred while submitting SLURM job: {e}")
            print("Return Code:", e.returncode)
            print("Error Output:", e.stderr)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Choosing which benchmarks to run")

    # Required arguments
    parser.add_argument("--hutch", type=str, help="Hutch to run benchmarks")

    # Hyperparameters for BayFAI
    parser.add_argument("--n_samples", type=int, default=20, help="Number of initial samples to draw")
    parser.add_argument("--n_iterations", type=int, default=80, help="Number of Bayesian optimization iterations")
    parser.add_argument("--max_rings", type=int, default=8, help="Max number of rings to consider")
    parser.add_argument("--prior", action="store_false", help="Flag to draw initial samples at random or from a gaussian prior defined by the center parameter")
    parser.add_argument("--beta", type=float, default=1.96, help="Exploration-exploitation trade-off parameter for UCB acquisition function")
    parser.add_argument("--step", type=int, default=5, help="Number of grid steps to refine around the BO best parameter for the gradient descent search")
    parser.add_argument("--smooth", action="store_true", help="Flag to smooth powder image or not")

    # Reproducibility
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    main(args)
