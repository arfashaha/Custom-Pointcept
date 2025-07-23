#!/bin/bash
#SBATCH -o /home/%u/slogs/sl_%A.out
#SBATCH -e /home/%u/slogs/sl_%A.out
#SBATCH -N 1	  # nodes requested
#SBATCH -n 1	  # tasks requested
#SBATCH --gres=gpu:2  # use 1 GPU
#SBATCH --mem=14000  # memory in Mb
#SBATCH --partition=PGR-Standard
#SBATCH -t 24:00:00  # time requested in hour:minute:seconds
#SBATCH --cpus-per-task=4  # number of cpus to use - there are 32 on each node.
set -e # fail fast

dt=$(date '+%d_%m_%y_%H_%M');
echo "I am job ${SLURM_JOB_ID}"
echo "I'm running on ${SLURM_JOB_NODELIST}"
echo "Job started at ${dt}"

# ====================
# Activate Anaconda environment
# ====================
source /home/${USER}/miniconda3/bin/activate pointcept

export PYTHONWARNINGS="ignore::FutureWarning"

# sh scripts/train.sh -g 2 -d scannet_custom -c semseg-spunet-v1m1-0-depth -n semseg-spunet-v1m1-0-depth
# sh scripts/train.sh -g 2 -d scannet_custom -c semseg-spunet-v1m1-0-width -n semseg-spunet-v1m1-0-width
sh scripts/train.sh -g 2 -d scannet_custom -c semseg-spunet-v1m1-0-combine -n semseg-spunet-v1m1-0-combine

echo ""
echo "============"
echo "job finished successfully"
dt=$(date '+%d/%m/%Y %H:%M:%S')
echo "Job finished: $dt"
