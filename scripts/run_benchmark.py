import os
import subprocess
from pathlib import Path
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
    hutch = args.hutch
    seed = args.seed
    max_rings = args.max_rings
    smooth = args.smooth
    yaml_folder = Path(f"../BayFAI/benchmark/yamls/{hutch}")
    yamls = yaml_folder.glob("*.yaml")
    work_dir = Path("../tests")
    test_folder_name = f"test_{hutch}_{seed}"
    test_folder = work_dir / test_folder_name
    test_folder.mkdir(parents=True, exist_ok=True)

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
                doc['BayFAI']['powder'] = f'/sdf/data/lcls/ds/prj/prjlute22/results/benchmarks/geom_opt/powder/{exp}_Run{run:0>4}.h5'
                doc['BayFAI']['preprocess'] = smooth
                doc['BayFAI']['bo_params']['n_samples'] = 20
                doc['BayFAI']['bo_params']['n_iterations'] = 80
                doc['BayFAI']['bo_params']['max_rings'] = max_rings
                doc['BayFAI']['bo_params']['seed'] = None

        with open(config_path, "w") as f:
            yaml.safe_dump_all(config, f)

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
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")

    # Hyperparameters for BayFAI
    parser.add_argument("--max_rings", type=int, default=8, help="Max number of rings to consider")
    parser.add_argument("--smooth", action="store_true", help="Flag to smooth powder image or not")
    
    args = parser.parse_args()
    main(args)
