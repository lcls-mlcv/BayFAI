#!/bin/bash
#SBATCH -p milano
#SBATCH --account=lcls:prjlute22
#SBATCH --job-name=get_node
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --mem=64G
#SBATCH --output=debug_bayfai.out
#SBATCH --error=debug_bayfai.err
#SBATCH --time=10:00:00

source /sdf/group/lcls/ds/ana/sw/conda1/manage/bin/psconda.sh 

sleep 2000000000000000000