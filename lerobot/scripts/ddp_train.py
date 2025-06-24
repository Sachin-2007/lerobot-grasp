#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Distributed Data Parallel (DDP) Training Script for LeRobot

This script enables distributed training across multiple GPUs using PyTorch's DistributedDataParallel.
It's specifically designed to work with the LeRobot training pipeline and supports:

- Multi-GPU training with gradient synchronization
- Episode-aware data sampling with proper distribution
- Distributed checkpointing and evaluation
- WandB logging from rank 0 only
- Proper cleanup and error handling

Usage:
    python ddp_train.py \
        --output_dir=outputs/train/ddp_act_aloha_insertion \
        --policy.type=act \
        --dataset.repo_id=lerobot/aloha_sim_insertion_human \
        --env.type=aloha \
        --env.task=AlohaInsertion-v0 \
        --batch_size=128 \
        --wandb.enable=true \
        --wandb.mode=online

Requirements:
    - At least 2 GPUs
    - PyTorch with distributed support
    - Gloo backend (recommended for Kaggle environment)

Note: This script uses mp.spawn() to create multiple processes for training.
Each process will be assigned to a different GPU (rank).
Batch size is per GPU (e.g., batch_size=128 with 2 GPUs = 256 total batch size).
"""

import sys

sys.path.insert(0, "/kaggle/working/myenv/lib/python3.11/site-packages")

import logging
import os
import time
from contextlib import nullcontext
from pprint import pformat
from typing import Any

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from termcolor import colored
from torch.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Optimizer
from torch.utils.data.distributed import DistributedSampler

from lerobot.common.datasets.factory import make_dataset
from lerobot.common.datasets.sampler import EpisodeAwareSampler
from lerobot.common.datasets.utils import cycle
from lerobot.common.envs.factory import make_env
from lerobot.common.optim.factory import make_optimizer_and_scheduler
from lerobot.common.policies.factory import make_policy
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.policies.utils import get_device_from_parameters
from lerobot.common.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.common.utils.random_utils import set_seed
from lerobot.common.utils.train_utils import (
    get_step_checkpoint_dir,
    get_step_identifier,
    load_training_state,
    save_checkpoint,
    update_last_checkpoint,
)
from lerobot.common.utils.utils import (
    format_big_number,
    get_safe_torch_device,
    has_method,
    init_logging,
)
from lerobot.common.utils.wandb_utils import WandBLogger
from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.scripts.eval import eval_policy


def setup(rank: int, world_size: int):
    """Initialize the distributed environment."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '5554'
    
    # Additional environment variables for better performance
    os.environ['NCCL_DEBUG'] = 'INFO'  # For debugging if needed
    os.environ['NCCL_SOCKET_IFNAME'] = 'lo'  # Use loopback for single-node training
    
    # Initialize the process group
    # Using gloo backend which is more reliable for CPU/mixed workloads
    # and works well in containerized environments like Kaggle
    try:
        dist.init_process_group("gloo", rank=rank, world_size=world_size)
        if rank == 0:
            print(f"Successfully initialized distributed training with {world_size} processes")
    except Exception as e:
        print(f"Failed to initialize distributed training: {e}")
        raise


def cleanup():
    """Clean up the distributed environment."""
    try:
        if dist.is_initialized():
            dist.destroy_process_group()
    except Exception as e:
        print(f"Error during cleanup: {e}")


def is_main_process(rank: int) -> bool:
    """Check if this is the main process (rank 0)."""
    return rank == 0


def wait_for_everyone():
    """Synchronize all processes."""
    if dist.is_initialized():
        dist.barrier()


class DistributedEpisodeAwareSampler(DistributedSampler):
    """
    A distributed sampler that works with episode-aware sampling for lerobot datasets.
    This sampler ensures that episode boundaries are respected while distributing data across ranks.
    """
    
    def __init__(
        self,
        episode_data_index,
        drop_n_last_frames: int = 0,
        num_replicas=None,
        rank=None,
        shuffle=True,
        seed=0,
    ):
        super().__init__(
            dataset=None,  # We'll handle dataset differently
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
            seed=seed,
        )
        self.episode_data_index = episode_data_index
        self.drop_n_last_frames = drop_n_last_frames
        
        # Create episode-aware indices
        self.valid_indices = []
        for start_index, end_index in zip(
            episode_data_index["from"],
            episode_data_index["to"],
            strict=True,
        ):
            episode_indices = list(range(start_index.item(), end_index.item() - drop_n_last_frames))
            self.valid_indices.extend(episode_indices)
        
        self.num_samples = len(self.valid_indices)
        # Distribute indices across ranks
        self.indices_per_rank = self.num_samples // self.num_replicas
        self.total_size = self.indices_per_rank * self.num_replicas
        
    def __iter__(self):
        if self.shuffle:
            # Generate random permutation with epoch-based seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.valid_indices), generator=g).tolist()
            indices = [self.valid_indices[i] for i in indices]
        else:
            indices = list(self.valid_indices)
        
        # Truncate to make it evenly divisible
        indices = indices[:self.total_size]
        
        # Subsample for this rank
        start_idx = self.rank * self.indices_per_rank
        end_idx = start_idx + self.indices_per_rank
        rank_indices = indices[start_idx:end_idx]
        
        return iter(rank_indices)
    
    def __len__(self):
        return self.indices_per_rank


def update_policy_ddp(
    train_metrics: MetricsTracker,
    policy: DDP,
    batch: Any,
    optimizer: Optimizer,
    grad_clip_norm: float,
    grad_scaler: GradScaler,
    lr_scheduler=None,
    use_amp: bool = False,
    lock=None,
) -> tuple[MetricsTracker, dict]:
    """
    Update policy with DDP support.
    Similar to the original update_policy but handles DDP-wrapped model.
    """
    start_time = time.perf_counter()
    device = get_device_from_parameters(policy.module)  # Use .module for DDP
    policy.train()
    with torch.autocast(device_type=device.type) if use_amp else nullcontext():
        loss, output_dict = policy(batch)  # DDP handles forward pass
        # TODO(rcadene): policy.unnormalize_outputs(out_dict)
    grad_scaler.scale(loss).backward()

    # Unscale the gradient of the optimizer's assigned params in-place **prior to gradient clipping**.
    grad_scaler.unscale_(optimizer)

    grad_norm = torch.nn.utils.clip_grad_norm_(
        policy.parameters(),
        grad_clip_norm,
        error_if_nonfinite=False,
    )

    # Optimizer's gradients are already unscaled, so scaler.step does not unscale them,
    # although it still skips optimizer.step() if the gradients contain infs or NaNs.
    with lock if lock is not None else nullcontext():
        grad_scaler.step(optimizer)
    # Updates the scale for next iteration.
    grad_scaler.update()

    optimizer.zero_grad()

    # Step through pytorch scheduler at every batch instead of epoch
    if lr_scheduler is not None:
        lr_scheduler.step()

    if has_method(policy.module, "update"):  # Use .module for DDP
        # To possibly update an internal buffer (for instance an Exponential Moving Average like in TDMPC).
        policy.module.update()

    train_metrics.loss = loss.item()
    train_metrics.grad_norm = grad_norm.item()
    train_metrics.lr = optimizer.param_groups[0]["lr"]
    train_metrics.update_s = time.perf_counter() - start_time
    return train_metrics, output_dict


def train_ddp(rank: int, world_size: int, cfg: TrainPipelineConfig):
    """
    Main distributed training function that runs on each process.
    """
    try:
        # Setup distributed training
        setup(rank, world_size)
        
        # Set device for this process
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)
        
        # Only log from rank 0 to avoid duplicate logs
        if is_main_process(rank):
            logging.info(pformat(cfg.to_dict()))
        
        # Initialize WandB logger only on rank 0
        wandb_logger = None
        if is_main_process(rank) and cfg.wandb.enable and cfg.wandb.project:
            wandb_logger = WandBLogger(cfg)
        elif is_main_process(rank):
            logging.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))

        if cfg.seed is not None:
            # Set different seed for each rank to ensure different data loading
            set_seed(cfg.seed + rank)

        # Enable optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True

        if is_main_process(rank):
            logging.info("Creating dataset")
        dataset = make_dataset(cfg)

        # Create environment used for evaluating checkpoints during training on simulation data.
        # Only on rank 0 to avoid multiple evaluation processes
        eval_env = None
        if is_main_process(rank) and cfg.eval_freq > 0 and cfg.env is not None:
            logging.info("Creating env")
            eval_env = make_env(cfg.env, n_envs=cfg.eval.batch_size, use_async_envs=cfg.eval.use_async_envs)

        if is_main_process(rank):
            logging.info("Creating policy")
        policy = make_policy(
            cfg=cfg.policy,
            ds_meta=dataset.meta,
        )
        
        # Move policy to the correct device before wrapping with DDP
        policy = policy.to(device)
        
        # Wrap the policy with DistributedDataParallel
        # find_unused_parameters=False for better performance if all parameters are used
        policy = DDP(
            policy, 
            device_ids=[rank],
            find_unused_parameters=False,  # Set to True if you have unused parameters
            broadcast_buffers=True,  # Synchronize buffers across ranks
        )

        if is_main_process(rank):
            logging.info("Creating optimizer and scheduler")
        optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy.module)  # Use .module for DDP
        grad_scaler = GradScaler(device.type, enabled=cfg.policy.use_amp)

        step = 0  # number of policy updates (forward + backward + optim)

        if cfg.resume:
            step, optimizer, lr_scheduler = load_training_state(cfg.checkpoint_path, optimizer, lr_scheduler)

        num_learnable_params = sum(p.numel() for p in policy.module.parameters() if p.requires_grad)
        num_total_params = sum(p.numel() for p in policy.module.parameters())

        if is_main_process(rank):
            logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
            if cfg.env is not None:
                logging.info(f"{cfg.env.task=}")
            logging.info(f"{cfg.steps=} ({format_big_number(cfg.steps)})")
            logging.info(f"{dataset.num_frames=} ({format_big_number(dataset.num_frames)})")
            logging.info(f"{dataset.num_episodes=}")
            logging.info(f"{num_learnable_params=} ({format_big_number(num_learnable_params)})")
            logging.info(f"{num_total_params=} ({format_big_number(num_total_params)})")
            logging.info(f"World size: {world_size}, Rank: {rank}")

        # Create distributed dataloader
        if hasattr(cfg.policy, "drop_n_last_frames"):
            # Use our custom distributed episode-aware sampler
            sampler = DistributedEpisodeAwareSampler(
                dataset.episode_data_index,
                drop_n_last_frames=cfg.policy.drop_n_last_frames,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                seed=cfg.seed if cfg.seed is not None else 0
            )
            shuffle = False
        else:
            sampler = DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                seed=cfg.seed if cfg.seed is not None else 0
            )
            shuffle = False

        dataloader = torch.utils.data.DataLoader(
            dataset,
            num_workers=cfg.num_workers,
            batch_size=cfg.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            pin_memory=device.type != "cpu",
            drop_last=True,  # Important for DDP to ensure consistent batch sizes
            persistent_workers=cfg.num_workers > 0,  # Keep workers alive between epochs
        )
        dl_iter = cycle(dataloader)

        policy.train()

        train_metrics = {
            "loss": AverageMeter("loss", ":.3f"),
            "grad_norm": AverageMeter("grdn", ":.3f"),
            "lr": AverageMeter("lr", ":0.1e"),
            "update_s": AverageMeter("updt_s", ":.3f"),
            "dataloading_s": AverageMeter("data_s", ":.3f"),
        }

        train_tracker = MetricsTracker(
            cfg.batch_size, dataset.num_frames, dataset.num_episodes, train_metrics, initial_step=step
        )

        if is_main_process(rank):
            logging.info("Start distributed offline training on a fixed dataset")
        
        for _ in range(step, cfg.steps):
            # Set epoch for the sampler to ensure proper shuffling across epochs
            if hasattr(sampler, 'set_epoch'):
                sampler.set_epoch(step)
                
            start_time = time.perf_counter()
            batch = next(dl_iter)
            train_tracker.dataloading_s = time.perf_counter() - start_time

            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device, non_blocking=True)

            train_tracker, output_dict = update_policy_ddp(
                train_tracker,
                policy,
                batch,
                optimizer,
                cfg.optimizer.grad_clip_norm,
                grad_scaler=grad_scaler,
                lr_scheduler=lr_scheduler,
                use_amp=cfg.policy.use_amp,
            )

            # Note: eval and checkpoint happens *after* the `step`th training update has completed, so we
            # increment `step` here.
            step += 1
            train_tracker.step()
            is_log_step = cfg.log_freq > 0 and step % cfg.log_freq == 0
            is_saving_step = step % cfg.save_freq == 0 or step == cfg.steps
            is_eval_step = cfg.eval_freq > 0 and step % cfg.eval_freq == 0

            # Only log from rank 0
            if is_main_process(rank) and is_log_step:
                # Synchronize metrics across all ranks
                wait_for_everyone()
                logging.info(train_tracker)
                if wandb_logger:
                    wandb_log_dict = train_tracker.to_dict()
                    if output_dict:
                        wandb_log_dict.update(output_dict)
                    wandb_logger.log_dict(wandb_log_dict, step)
                train_tracker.reset_averages()

            # Only save checkpoints from rank 0
            if is_main_process(rank) and cfg.save_checkpoint and is_saving_step:
                # Synchronize before saving
                wait_for_everyone()
                logging.info(f"Checkpoint policy after step {step}")
                checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, step)
                save_checkpoint(checkpoint_dir, step, cfg, policy.module, optimizer, lr_scheduler)  # Use .module
                update_last_checkpoint(checkpoint_dir)
                if wandb_logger:
                    wandb_logger.log_policy(checkpoint_dir)

            # Only evaluate on rank 0
            if is_main_process(rank) and cfg.env and is_eval_step:
                # Synchronize before evaluation
                wait_for_everyone()
                step_id = get_step_identifier(step, cfg.steps)
                logging.info(f"Eval policy at step {step}")
                with (
                    torch.no_grad(),
                    torch.autocast(device_type=device.type) if cfg.policy.use_amp else nullcontext(),
                ):
                    eval_info = eval_policy(
                        eval_env,
                        policy.module,  # Use .module for evaluation
                        cfg.eval.n_episodes,
                        videos_dir=cfg.output_dir / "eval" / f"videos_step_{step_id}",
                        max_episodes_rendered=4,
                        start_seed=cfg.seed,
                    )

                eval_metrics = {
                    "avg_sum_reward": AverageMeter("∑rwrd", ":.3f"),
                    "pc_success": AverageMeter("success", ":.1f"),
                    "eval_s": AverageMeter("eval_s", ":.3f"),
                }
                eval_tracker = MetricsTracker(
                    cfg.batch_size, dataset.num_frames, dataset.num_episodes, eval_metrics, initial_step=step
                )
                eval_tracker.eval_s = eval_info["aggregated"].pop("eval_s")
                eval_tracker.avg_sum_reward = eval_info["aggregated"].pop("avg_sum_reward")
                eval_tracker.pc_success = eval_info["aggregated"].pop("pc_success")
                logging.info(eval_tracker)
                if wandb_logger:
                    wandb_log_dict = {**eval_tracker.to_dict(), **eval_info}
                    wandb_logger.log_dict(wandb_log_dict, step, mode="eval")
                    wandb_logger.log_video(eval_info["video_paths"][0], step, mode="eval")

            # Synchronize all processes at the end of each step
            wait_for_everyone()

        if is_main_process(rank):
            if eval_env:
                eval_env.close()
            logging.info("End of distributed training")

    except Exception as e:
        if is_main_process(rank):
            logging.error(f"Training failed with error: {e}")
        raise
    finally:
        # Cleanup distributed training
        cleanup()


@parser.wrap()
def train(cfg: TrainPipelineConfig):
    """
    Main entry point for distributed training.
    This function spawns multiple processes for DDP training.
    """
    cfg.validate()
    
    # Check if we have multiple GPUs available
    n_gpus = torch.cuda.device_count()
    print(f"Total GPUs available: {n_gpus}")
    
    if n_gpus < 2:
        print(f"DDP training requires at least 2 GPUs, but only {n_gpus} available.")
        print("Falling back to single GPU training...")
        # You could import and call the original train function here
        raise RuntimeError("DDP training requires at least 2 GPUs")
    
    world_size = n_gpus
    print(f"Starting DDP training with {world_size} processes")
    
    # Spawn processes for distributed training
    mp.spawn(
        train_ddp,
        args=(world_size, cfg),
        nprocs=world_size,
        join=True
    )


if __name__ == "__main__":
    # Initialize logging only for the main process
    init_logging()
    train()
