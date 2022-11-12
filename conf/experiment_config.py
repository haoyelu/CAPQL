from omegaconf import DictConfig, OmegaConf
from omegaconf import MISSING
from hydra.core.config_store import ConfigStore
from hydra import compose, initialize
from hydra.conf import HydraConf
from typing import Any
from dataclasses import dataclass, field

@dataclass
class ModelConfig:
    type: str = 'CAPQL'
    hidden_size: int = 256

@dataclass
class TrainConfig:
    eval: bool = True
    gamma: float = 0.99
    tau: float = 0.005
    lr: float = 0.0003
    alpha: float = 0.2
    seed: int = 123456
    batch_size: int = 256
    num_steps: int = 10000001
    updates_per_step: int = 1
    start_steps: int = 10000
    delta: float = 0.1
    target_update_interval: int = 1
    replay_size: int = 1000000

@dataclass
class WeightSamplerConfig:
    control_angle: bool = True
    angle: float = 22.5

@dataclass
class GPUConfig:
    cuda: bool = True

@dataclass
class ModelTrainConfig:
    model: ModelConfig = ModelConfig()
    training: TrainConfig = TrainConfig()
    weight_sampler: Any = WeightSamplerConfig()
    gpu: GPUConfig = GPUConfig()
    name: str = "HopperM-v0"
    hydra: HydraConf = HydraConf()
