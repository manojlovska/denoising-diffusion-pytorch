#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4-00:00:00
#SBATCH --partition=gpu
#SBATCH --job-name=ddpm-pytorch
#SBATCH --output=/d/hpc/projects/mixbai_fe/DDPM/denoising-diffusion-pytorch/outputs/train_log_%j.log
#SBATCH --export=WANDB_API_KEY
#SBATCH --export=HTTPS_PROXY
#SBATCH --export=https_proxy

source /ceph/grid/home/am6417/miniconda3/etc/profile.d/conda.sh
conda activate env_ddpm

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=7200

accelerate launch --config_file default_config.yaml train_ddpm.py