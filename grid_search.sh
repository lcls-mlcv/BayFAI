#!/bin/bash
#SBATCH -p milano
#SBATCH --account=lcls:prjlute22
#SBATCH --job-name=grid_search
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --mem=64G
#SBATCH --output=grid_search.out
#SBATCH --error=grid_search.err
#SBATCH --time=24:00:00

source /sdf/group/lcls/ds/ana/sw/conda1/manage/bin/psconda.sh 

PYTHON=python  # or python3 if needed
SCRIPT="/sdf/home/l/lconreux/exp/prjlute22/results/benchmarks/geom_opt/BayFAI/scripts/grid_search.py"

$PYTHON "$SCRIPT" \
    --exp "mecl1048423" \
    --run "89" \
    --detname "Epix10kaQuad2" \
    --in_file "/sdf/data/lcls/ds/mec/mecl1048423/calib/Epix10kaQuad::CalibV1/MecTargetChamber.0:Epix10kaQuad.2/geometry/0-end.data" \
    --out_dir "/sdf/data/lcls/ds/prj/prjlute22/results/benchmarks/geom_opt/geom/mecl1048423/" \
    --powder_path "/sdf/data/lcls/ds/mec/mecl1048423/results/bayfai/smd_output/mecl1048423_Run0089.h5" \
    --smooth \
    --threshold "95" \
    --calibrant "CeO2" \
    --wavelength "6.80935036100377e-11" \
    --fixed "[\"rot3\"]" \
    --center "{\"dist\": 0.2, \"poni1\": 0.0, \"poni2\": 0.0, \"rot1\": 0.0, \"rot2\": 0.0, \"rot3\": 0.0}" \
    --bounds "{\"dist\": [-0.05, 0.05], \"poni1\": [-0.005, 0.005], \"poni2\": [-0.005, 0.005], \"rot1\": [-1.6, 1.6], \"rot2\": [-1.6, 1.6], \"rot3\": [-1.6, 1.6]}" \
    --resolution "{\"dist\": 0.005, \"poni1\": 0.0001, \"poni2\": 0.0001, \"rot1\": 0.2, \"rot2\": 0.2, \"rot3\": 0.2}" \
    --n_init "100" \
    --n_iter "100" \
    --max_rings "6" \
    --rtol "1e-3" \
    --beta "1.96" \
    --prior \
    --seed "42"
