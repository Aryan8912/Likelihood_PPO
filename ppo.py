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

    output_dir: str = "models./ppo_tldr_pythia_1_4b"
    lora_alpha: int = 2048
    lora_dropout: float = 0.0
    task: TaskHParams = field(default_factory=TaskHParams)
    reward: RewardHParams = field(default_factory=RewardHParams)
    ppo: PpoHparams = field(default_factory=PpoHParams)

    def configure_dropout(model_config, dropout_layer_keys, dropout):
        if dropout is not None:
            for key in dropout_layer_keys:
                if hasattr(model_config, key):
                    print(f"Setting model_config.{key} to {dropout}")
                    setattr(model_config, key, dropout)

    
    def print_rich_table(title: str, df: pd.DataFrame, console: Console) -> Table:
        table = Table(show_lines = True)
        for column is df.columns:
            table.add_column(column)
        for _, row in df.iterrows():
            table.add_row(*row.astype(str).tolist())
        
        console.rule(f"[bold red] {title}")
        console.print(table)

    def layer_init(layer, std=np.sqrt(2), bias_const=0.0)
        torch.nn.init.normal_(layer.weight, std=std)
        torch.nn.init.constant_(layer.bias, val=bias_const)
        return layer
    
    class AdaptiveKLController:
        def __init__(self, init_kl_coef: float, hparams: AdaptiveKLParams):
            self.value = init_kl_coef
            self.hparams = hparams

        def update(self, current, n_steps):
            target = self.hparams.target
            proportional_error = np.clip(current / target -1, -0.2, 0.2)
            mult = 1 + proportional_error * n_steps / self.hparams.horizon
            self.value *= mult

        def whiten(values, shift_mean=True):
            
            mean, var = torch.mean(values), torch.var(values, unbiased=False)
            whitened = (values - mean) * torch.rsqrt(var + 1e-8)
            if not shift_mean:
                whitened += mean
            
            return whitened
        
        class ScalarModelConfig(PretrainedConfig):
            def __init__(
                    self,
                    base_model: str = "EleutherAI/pythia-160m",
                    base_config: PretrainedConfig = AutoConfig.from_pretrained("EleutherAI/pythia-160m"),
                    hidden_size: int = 768,
                    bias: float = 0.0,
                    **kwargs,
            ):
                super().__init__(**kwargs)
                self.base_model = base_model
                self.base_config = base_config
                self.hidden_size = hidden_size
                self.bias = bias

        class ScalarModel(PreTrainedModel):
            config_class = ScalarModelConfig

            def __init__(self, config: ScalarModelConfig):
                super().__init__(config)
                self.config = config
                self.lm_backbone = AutoModel.from_pretrained(
                    config.base_model,
                    config=self.config.base_config,
                    trust_remote_code=True,
                )
                self.scalar_head = layer_init(
                    nn.Linear(self.config.hidden_size, 1),
                    std = 1 / np.sqrt(self.config.hidden_size + 1)
                )
            
            def forward(self, **kwargs):
                output = self.lm_backbone(**kwargs)
                reward = self.scalar_head(output.last_hidden_state[-1]) - self.config.bias
                return reward
            
            def get_reward(model, query_responses, tokenizer, context_length):
                attention_mask = query_response != tokenizer.pad_token_id
                input_ids = torch.masked_fill(query_responses, ~attention_mask, 0)
                reward_logits = model(
                    input_ids = input_ids,
                    attention_mask = attention_mask,
                    return_dict = True,
                    output_hidden_states = True,
                )
                sequence_lengths = first_true_indices(query_responses[:, context_length:] == tokenizer.pad_token_id) - 1 + context_length

                return (
                    reward_logits,
                    reward_logits[torch.arange(reward_logits.size(0), device=reward_logits.device), sequence_lenghts].squeeze(-1),
                    sequence_lengths,
                )
            
# 274