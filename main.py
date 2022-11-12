"""Main training script"""
import os
import hydra
import warnings

from typing import Any, Dict, List, Optional, Sequence
from training_pipeline import *
from hydra.core.config_store import ConfigStore
from conf.experiment_config import ModelTrainConfig
from omegaconf import DictConfig, OmegaConf

import cloudpickle
import conf
cloudpickle.register_pickle_by_value(conf)
import logging
from envs.env import *

log = logging.getLogger(__name__)

class Taskfunction():
    def __init__(self):
        self.model = None
        self.name = 'Taskfunction_name_untitled'

    def __call__(self, config: DictConfig, dir: str = None):
        print(f'initialize ... ')
        register_env()
        print(f'making env')
        try:
            env = gym.make(config.name)
        except Exception as inst:
            print(f'making failed')
            print(type(inst))
            print(inst.args) 
            print(inst)
            
        print(f'init model')
        self.model = TrainingProcess(config, env)

        print('Start running')
        self.model.__call__()

if __name__ == "__main__":
    cs = ConfigStore.instance()
    cs.store( name='config', node=ModelTrainConfig)
    trainfn = Taskfunction()
    app = hydra.main(config_path=None, config_name='config', version_base=None)(trainfn.__call__)
    app()

    