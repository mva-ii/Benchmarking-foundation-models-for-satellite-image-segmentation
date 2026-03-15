#!/bin/bash -l
# (submit.sh)

# SLURM SUBMIT SCRIPT
#SBATCH --partition=gpu_test
#SBATCH --nodes=1             # This needs to match Trainer(num_nodes=...)
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=10
#SBATCH --ntasks-per-node=4   # This needs to match Trainer(devices=...)
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --error=%j.out
#SBATCH --output=%j.out

module purge

# debugging flags (optional)
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1
export CUDA_LAUNCH_BLOCKING=1
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_SOCKET_IFNAME=^docker0,lo
export WANDB_CACHE_DIR=$WORKDIR/wandb_cache
export WANDB_DIR=$WORKDIR/wandb
export WANDB_DATA_DIR=$WORKDIR/wandb_data

# on your cluster you might need these:
# set the network interface

# might need the latest CUDA
# module load NCCL/2.4.7-1-cuda.10.0

echo "Activating environment from $ENVIRONMENT_ROOT"
echo "Using config file $CONFIG_FILE"
echo "PASTIS-R root: $PASTIS_R_ROOT"
echo "Embedding root: $EMBEDDING_ROOT"

source $ENVIRONMENT_ROOT/bin/activate

# run script from above
srun python -u scripts/runme.py fit -c $CONFIG_FILE --data.init_args.pastis_r_root=$PASTIS_R_ROOT --data.init_args.embedding_root=$EMBEDDING_ROOT