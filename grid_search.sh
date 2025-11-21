#!/bin/bash
#SBATCH -p milano
#SBATCH --account=lcls:prjlute22
#SBATCH --job-name=grid_search
#SBATCH -N 1
#SBATCH --ntasks=10
#SBATCH --mem=128G
#SBATCH --output=grid_inty.out
#SBATCH --error=grid_inty.err
#SBATCH --time=48:00:00

source /sdf/group/lcls/ds/ana/sw/conda1/manage/bin/psconda.sh 

SCRIPT="/sdf/home/l/lconreux/exp/prjlute22/results/benchmarks/geom_opt/BayFAI/scripts/grid_search.py"

mpirun -np ${SLURM_NTASKS} python "$SCRIPT" \
    --exp "mfx100824024" \
    --run 5 \
    --detname "epix10k2M" \
    --in_file "/sdf/data/lcls/ds/prj/prjlute22/results/benchmarks/geom_opt/geom/mfx100824024/0-end.data" \
    --out_dir "/sdf/data/lcls/ds/prj/prjlute22/results/benchmarks/geom_opt/grid_search/mfx100824024/" \
    --powder_path "/sdf/data/lcls/ds/prj/prjlute22/results/benchmarks/geom_opt/powder/mfx100824024_Run0005.h5" \
    --calibrant "LaB6" \
    --wavelength "1.4442606539534077e-10" \
    --fixed '["rot1", "rot2", "rot3"]' \
    --center '{"dist": 0.1, "poni1": 0.0, "poni2": 0.0, "rot1": 0.0, "rot2": 0.0, "rot3": 0.0}' \
    --bounds '{"dist": [-0.05, 0.05], "poni1": [-0.01, 0.01], "poni2": [-0.01, 0.01], "rot1": [-1.6, 1.6], "rot2": [-1.6, 1.6], "rot3": [-1.6, 1.6]}' \
    --resolutions '{"dist": 0.005, "poni1": 0.0002, "poni2": 0.0002, "rot1": 0.4, "rot2": 0.4, "rot3": 0.2}' \
    --max_rings 6 \
    --score "powder_residual"
