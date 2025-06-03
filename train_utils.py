import os
import torch
from tqdm import tqdm
from loguru import logger
import wandb
import yaml
import numpy as np
import re
import psutil
import datetime

import pynvml
import time
import threading
import torch.distributed as dist

from accelerate import Accelerator
import torchvision
from torchvision.utils import make_grid
from torch.optim import Adam
from torch.utils.data import DataLoader

from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer

# os.environ["NCCL_P2P_DISABLE"] = "1"
# os.environ["NCCL_IB_DISABLE"] = "1"
# os.environ["TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC"] = "7200"

# print(f"NCCL_P2P_DISABLE {os.environ['NCCL_P2P_DISABLE']}")
# print(f"NCCL_IB_DISABLE {os.environ['NCCL_IB_DISABLE']}")
# print(f"TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC {os.environ['TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC']}")


# # Initialize NVML (for NVIDIA GPUs)
# pynvml.nvmlInit()
# gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # change index if you have multiple GPUs

# def log_system_metrics(log_interval=5, log_file="system_metrics.log"):
#     with open(log_file, "a") as f:
#         while True:
#             # Get CPU utilization
#             cpu_usage = psutil.cpu_percent(interval=1)
#             # Get RAM usage
#             mem = psutil.virtual_memory()
#             ram_used = mem.used / (1024**3)  # in GB
#             ram_total = mem.total / (1024**3)  # in GB
            
#             # Get GPU metrics
#             gpu_util = pynvml.nvmlDeviceGetUtilizationRates(gpu_handle).gpu
#             gpu_mem = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle)
#             vram_used = gpu_mem.used / (1024**3)  # in GB
#             vram_total = gpu_mem.total / (1024**3)  # in GB
            
#             # Build log message
#             log_message = (f"{time.strftime('%Y-%m-%d %H:%M:%S')} | CPU: {cpu_usage}% | "
#                            f"RAM: {ram_used:.2f}/{ram_total:.2f} GB | "
#                            f"GPU Util: {gpu_util}% | VRAM: {vram_used:.2f}/{vram_total:.2f} GB\n")
#             f.write(log_message)
#             f.flush()  # ensure it's written to disk
#             time.sleep(log_interval)
            
# # Start logging in a background thread
# threading.Thread(target=log_system_metrics, daemon=True).start()

class Train:
    def __init__(self, images_folder, results_folder, project_name, milestone=None):
        self.images_folder = images_folder
        self.results_folder = results_folder
        self.project_name = project_name
        self.milestone = milestone

        super().__init__

    def train(self):
        self.training_setup()
        try:
            self.train_wandb()
        except Exception:
            raise
        finally:
            self.after_train()

    def training_setup(self):
        logger.info("Preparing for train ...")

        # Model and diffusion process
        self.model = Unet(
            dim = 64,
            dim_mults = (1, 2, 4, 8),
            flash_attn = False
        )

        self.diffusion = GaussianDiffusion(
            self.model,
            image_size = 128,
            timesteps = 1000,           # number of steps
            sampling_timesteps = 250    # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
        )

        # Save folder
        self.filename = os.path.join(self.results_folder, self.project_name)

        # Get trainer
        self.trainer = Trainer(
            self.diffusion,
            folder=self.images_folder,
            train_batch_size = 32,
            train_lr = 8e-5,
            train_num_steps = 600000,         # total training steps
            gradient_accumulate_every = 2,    # gradient accumulation steps
            ema_decay = 0.995,                # exponential moving average decay
            amp = True,                       # turn on mixed precision
            calculate_fid = True,              # whether to calculate fid during training
            results_folder=self.filename,
            log_wandb=False,
            num_fid_samples=10000,
            save_and_sample_every = 25000,
            milestone=self.milestone
        )

        if self.milestone is not None:
            self.trainer.load(self.milestone)

    def train_wandb(self):
        self.trainer.train_wandb()
    
    def after_train(self):
        logger.info("Training of the model is done.")



