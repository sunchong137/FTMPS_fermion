#!/bin/bash
# parallel job using 20 processors. and runs for 1 hours (max) 
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
##SBATCH -c 20
#SBATCH -t 10:00:00
#SBATCH --partition=smallmem
##SBATCH -C ivy
##SBATCH --mem=100000
# sends mail when process begins, and
# when it ends. Make sure you define your email
# address.

#SBATCH --job-name=ftmpo
##SBATCH --mail-type=begin
##SBATCH --mail-type=end
##SBATCH --mail-user=sunchong137@gmail.com

srun hostname |sort
ulimit -l unlimited
source /home/sunchong/.modulerc
export PYTHONPATH=/home/sunchong/work:$PYTHONPATH
export OMP_NUM_THREADS=1
export SCRATCHDIR="/scratch/local/sunchong"
##srun ../mpo_ancilla_hub hubbard_input
srun ../mpo_ancilla_hub hubNUM

