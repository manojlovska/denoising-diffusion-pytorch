#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --time=7-00:00:00
#SBATCH --partition=compute
#SBATCH --job-name=ddpm-pytorch
#SBATCH --output=/ceph/grid/home/am6417/Projects/DDPM/denoising-diffusion-pytorch/outputs/train_log_%j.log
#SBATCH --export=WANDB_API_KEY
#SBATCH --export=HTTPS_PROXY
#SBATCH --export=https_proxy

source /ceph/grid/home/am6417/miniconda3/etc/profile.d/conda.sh
conda activate env_ddpm

export NCCL_DEBUG=INFO
# export NCCL_P2P_DISABLE=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=18000

echo $NCCL_P2P_DISABLE
echo $NCCL_IB_DISABLE
echo $TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC

# export NCCL_DEBUG=INFO
# export NCCL_P2P_DISABLE=1
# export NCCL_IB_DISABLE=1
# export NCCL_ASYNC_ERROR_HANDLING=1
# export NCCL_SOCKET_IFNAME=eth0
# export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=7200
# export TORCH_DISTRIBUTED_DEBUG=DETAIL

# export NCCL_IB_DISABLE=1
# export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=7200

accelerate launch --config_file default_config.yaml train_ddpm.py