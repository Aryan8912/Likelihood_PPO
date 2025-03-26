import os 
import random
import time
from dataclasses import asdict, dataclass, field
from types import SimpleNamespace
from typing import List, Literal, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
import wandb
import deepseed
from accelerate import Accelerator
from accelerate.state import AcceleratorState
from accelerate.utils import gather_object
from datasets import load_dataset
from rich.console import Console
from rich.table import Table
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    PretrainingedConfig,
    PreTrainedModel,
)
from peft import get_peft_model, LoraConfig

@dataclass
class AdaptiveKLParams:
    target: float = 6.0
    horizon: int = 10000

@dataclass 
class PpoParams:
    num_updates: tyro.conf.Suppress[int] = None
    noptepochs: int = 4
    vf_coef: float = 0.1
    cliprange: float = 0.2
    cliprange_value: float = 0.2
    gamma: float = 1
    lam: float = 0.95
    whiten_rewards: bool = True

@dataclass
class TaskHParams:
    query_length: int = 512
    query_dataset: str = "cleanrl/summarize_from_feedback_tldr_3_filtered_oai_preprocessing_1704563162"

    response_length: int = 53

    truncate_token = Literal["eos"] = "eos"
    truncate_token_id: Optional[int] = None
    penalty_reward_value: int = -1

    temperature: float = 0.7

@dataclass
class Args:
    exp_name: str = "pythia_ppo"
    seed: int = 555134
    track: bool = True
    wandb_project_name: str = "tldr_summarize_pythia"
    cuda: bool = True
    run_name: Optional[str] = None
    push_to_hub: bool = False
    hf_entity: str = ""
    deepspeed: bool = True
    print_sample_output_freq: int = 100
    run_eval: bool = True

    eps: float = 1e-5
    lr: float = 3e-6
    optimizer: Literal["adam", "adamw"] = "adamw"
    scheduler: str = "cosine"
    warm_up_steps: int = 0
    
    num_train_epochs: int = 1
    gradient_accumulation_steps: int = 32

    per_device_eval_batch_size: int = 2
    total_episodes: int = 1000000

    world_size: Optional[int] = 8

    batch_size: Optional[int] = 512

    local_rollout_forward_batch_size: int = 32

    local_rollout_forward_batch_size: int = 32

    local_batch_size: Optional[int] = 128

    base_model: str = "model/sft_tldr_pythia_1_4b"
    
    offload: bool = False

    reward_model_path: str = "model/rm_sft_tldr_pythia_1_4b"

    sft_model_path: str = "model/sft_tldr_pythia_1_4b"

    dropout_layer_keys: List[str] = field(
        default_factory=lambda: ["attn_pdrop", "embd_pdrop", "resid_pdrop", "summary_first_dropout"]
    )

    # 155