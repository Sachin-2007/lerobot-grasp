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

import sys
sys.path.insert(0, "/kaggle/working/myenv/lib/python3.11/site-packages")

import logging
import os
import sys
import time
from contextlib import nullcontext
from pprint import pformat
from typing import Any

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.amp import GradScaler
from torch.optim import Optimizer

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


def setup_ddp(rank, world_size):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup_ddp():
    dist.destroy_process_group()

def update_policy_ddp(
    train_metrics: MetricsTracker,
    policy: PreTrainedPolicy,
    batch: Any,
    optimizer: Optimizer,
    grad_clip_norm: float,
    grad_scaler: GradScaler,
    lr_scheduler=None,
    use_amp: bool = False,
    lock=None,
):
    start_time = time.perf_counter()
    device = get_device_from_parameters(policy)
    policy.train()
    with torch.autocast(device_type=device.type) if use_amp else nullcontext():
        loss, output_dict = policy.forward(batch)
    grad_scaler.scale(loss).backward()
    grad_scaler.unscale_(optimizer)
    grad_norm = torch.nn.utils.clip_grad_norm_(
        policy.parameters(),
        grad_clip_norm,
        error_if_nonfinite=False,
    )
    with lock if lock is not None else nullcontext():
        grad_scaler.step(optimizer)
    grad_scaler.update()
    optimizer.zero_grad()
    if lr_scheduler is not None:
        lr_scheduler.step()
    if has_method(policy, "update"):
        policy.update()
    train_metrics.loss = loss.item()
    train_metrics.grad_norm = grad_norm.item()
    train_metrics.lr = optimizer.param_groups[0]["lr"]
    train_metrics.update_s = time.perf_counter() - start_time
    return train_metrics, output_dict

def ddp_train_worker(rank, world_size, cfg: TrainPipelineConfig):
    try:
        setup_ddp(rank, world_size)
        if rank == 0:
            init_logging()
        if cfg.seed is not None:
            set_seed(cfg.seed + rank)
        device = torch.device(f"cuda:{rank}")
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        if rank == 0:
            logging.info(pformat(cfg.to_dict()))
        dataset = make_dataset(cfg)
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            num_workers=cfg.num_workers,
            batch_size=cfg.batch_size // world_size,
            sampler=sampler,
            pin_memory=True,
            drop_last=False,
        )
        policy = make_policy(cfg=cfg.policy, ds_meta=dataset.meta).to(device)
        policy = DDP(policy, device_ids=[rank], output_device=rank, find_unused_parameters=True)
        optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)
        grad_scaler = GradScaler(device.type, enabled=cfg.policy.use_amp)
        step = 0
        if cfg.resume:
            step, optimizer, lr_scheduler = load_training_state(cfg.checkpoint_path, optimizer, lr_scheduler)
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
        if rank == 0:
            logging.info("Start DDP offline training on a fixed dataset")
        for _ in range(step, cfg.steps):
            sampler.set_epoch(_)
            start_time = time.perf_counter()
            try:
                batch = next(iter(dataloader))
            except Exception as e:
                if rank == 0:
                    logging.error(f"Error loading batch: {e}")
                continue
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
            step += 1
            train_tracker.step()
            is_log_step = cfg.log_freq > 0 and step % cfg.log_freq == 0
            is_saving_step = step % cfg.save_freq == 0 or step == cfg.steps
            if is_log_step and rank == 0:
                logging.info(train_tracker)
                train_tracker.reset_averages()
            if cfg.save_checkpoint and is_saving_step and rank == 0:
                logging.info(f"Checkpoint policy after step {step}")
                checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, step)
                save_checkpoint(checkpoint_dir, step, cfg, policy.module, optimizer, lr_scheduler)
                update_last_checkpoint(checkpoint_dir)
        cleanup_ddp()
        if rank == 0:
            logging.info("End of DDP training")
    except Exception as e:
        if rank == 0:
            logging.error(f"DDP training failed: {e}")
        cleanup_ddp()
        sys.exit(1)

@parser.wrap()
def ddp_train(cfg: TrainPipelineConfig):
    world_size = torch.cuda.device_count()
    if world_size < 2:
        logging.error("DDP training requires at least 2 GPUs.")
        sys.exit(1)
    mp.spawn(ddp_train_worker, args=(world_size, cfg), nprocs=world_size, join=True)

if __name__ == "__main__":
    ddp_train() 