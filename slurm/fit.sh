#!/bin/bash -l
# (submit.sh)

# SLURM SUBMIT SCRIPT
#SBATCH --partition=gpu_test
#SBATCH --nodes=4             # This needs to match Trainer(num_nodes=...)
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=2
#SBATCH --ntasks-per-node=4   # This needs to match Trainer(devices=...)
#SBATCH --mem=128G
#SBATCH --time=00:10:00

# debugging flags (optional)
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1
export CUDA_LAUNCH_BLOCKING=1

# on your cluster you might need these:
# set the network interface
# export NCCL_SOCKET_IFNAME=^docker0,lo

# might need the latest CUDA
# module load NCCL/2.4.7-1-cuda.10.0

# run script from above
srun uv run scripts/runme.py -c $CONFIG_FILE --data.init_args.pastis_r_root=$PASTIS_R_ROOT --data.init_args.embedding_root=$EMBEDDING_ROOT