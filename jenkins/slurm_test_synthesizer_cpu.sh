#!/bin/bash

#SBATCH --partition build
#SBATCH --time 00-00:15:00
#SBATCH --qos gpu
#SBATCH --gres gpu:1
#SBATCH --mem 40G
#SBATCH --cpus-per-task 4

set -e

#module load gcc/8.4.0-cuda
#module load cuda/10.2.89
#module load intel-vtune
#module list

# From Nicolas Ricart, as the Vtune from the stack does not 
# work with python 3.7
source /work/scitas-ge/richart/test_stacks/syrah/v1/opt/spack/linux-rhel7-skylake_avx512/gcc-8.4.0/intel-oneapi-vtune-2021.6.0-34ym22fgautykbgmg5hhgkiwrvbwfvko/setvars.sh

source pypeline.sh --no_shell
which python
python -V
pip show pypeline
echo
pwd
hostname
echo

env | grep SLURM

#EO: numexpr: check env and tidy up this
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export VECLIB_MAXIMUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

# TEST_DIR defined via Jenkins, so default to local when outside
OUTPUT_DIR=${TEST_DIR:-.}
echo OUTPUT_DIR = $OUTPUT_DIR

# Script to be run
PY_SCRIPT="./benchmarking/test_synthesizer_cpu.py"

# Timing
time python $PY_SCRIPT
echo; echo

# cProfile
time python -m cProfile -o $OUTPUT_DIR/cProfile.out $PY_SCRIPT
echo; echo

# Intel VTune Amplifier
amplxe-cl -collect hotspots -strategy ldconfig:notrace:notrace -result-dir=$OUTPUT_DIR -- ~/miniconda3/envs/pypeline/bin/python $PY_SCRIPT
echo; echo

ls -rtl $OUTPUT_DIR