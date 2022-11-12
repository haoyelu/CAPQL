import argparse
import datetime
import gym
import numpy as np
import itertools
import torch
from torch.utils.tensorboard import SummaryWriter
from utils.replay_memory import ReplayMemory
import time
import json, pickle, os
import hydra
import pickle
import logging
import shutil
from omegaconf import OmegaConf
log = logging.getLogger(__name__)

class TrainingProcess:
    def __init__(self, config, env, agent=None, sampler=None, memory=None, misc=None):
        self.config = config
        logging.info(f'config set')
        # init env
        self.env = env

        # init weight sampler
        if  config.weight_sampler.angle > 0: 
            from utils.utils import Weight_Sampler_angle as WS
            angle_d = config.weight_sampler.angle

            # convert angle in degree to radius      
            angle = torch.pi * (config.weight_sampler.angle / 180)
            self.sampler = WS(self.env.rwd_dim, angle)
            logging.info(f'Using weight sampler with controlled deg {angle_d}')
        else:
            from utils.utils import Weight_Sampler_pos as WS
            self.sampler = WS(self.env.rwd_dim)
            angle_d = 'PosW'
            logging.info(f'Using weight sampler with positive entries')

            # set random seeds
            self.env.seed(config.training.seed)
            self.env.action_space.seed(config.training.seed)
            torch.manual_seed(config.training.seed)
            np.random.seed(config.training.seed)
 
        # init agent
        if config.model.type == "CAPQL":
            from models import capql
            self.agent = capql.CAPQL(self.env.observation_space.shape[0], self.env.action_space, self.env.rwd_dim, config)
        else: 
            from models import qenv_ctn
            self.agent = qenv_ctn.QENV_CTN(self.env.observation_space.shape[0], self.env.action_space, self.env.rwd_dim, config)


        logging.info(f'Agent initialized')
       
        # set replay buffer memory
        self.memory = ReplayMemory(config.training.replay_size, config.training.seed)
        logging.info(f'Replay buffer initialized')

        # set misc
        self.total_numsteps = 0
        self.updates = 0
        self.epi_start = 1
        self.i_episode = 1

        # set the dir to checkpoint/tensorboard
        if config.model.type == 'CAPQL':
            self.dir_checkpoint = 'runs/{}/{}_{}_Angle_{}_batch_{}_{}_alpha_{}_tau_{}_seed_{}'.format(
                                config.name, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), config.model.type, 
                                str(angle_d), str(config.training.batch_size),str(config.model.hidden_size), 
                                config.training.alpha, config.training.tau,config.training.seed
                                )
        else:
            self.dir_checkpoint = 'runs/{}/{}_{}_Angle_{}_batch_{}_{}_delta_{}_alpha_{}_tau_{}_seed_{}'.format(
                                config.name, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), config.model.type, 
                                str(angle_d), str(config.training.batch_size),str(config.model.hidden_size), 
                                config.training.delta, config.training.alpha, config.training.tau,config.training.seed
                        )
        logging.info(f'misc initialized')
    
        if not os.path.exists(f'{self.dir_checkpoint}'):
            logging.info(f'{self.dir_checkpoint} does not exsit; create')
            os.makedirs(f'{self.dir_checkpoint}')
        else:
            logging.info(f'{self.dir_checkpoint} exsits;')

        # init tensorboard writer
        self.writer = SummaryWriter(self.dir_checkpoint)
 
    def __call__(self):
        while True:
            episode_reward = 0
            episode_steps = 0
            done = False
            state = self.env.reset()
            total_numsteps = self.total_numsteps
            updates = self.updates
            epi_start = self.epi_start
            i_episode = self.i_episode

            while not done:
                # pick a weight
                w = self.sampler.sample(1).view(-1)

                if self.config.training.start_steps > total_numsteps:
                    action = self.env.action_space.sample()  # Sample random action
                else:
                    action = self.agent.select_action(state, w)  # Sample action from policy
                start = time.time()
                
                if len(self.memory) > self.config.training.batch_size:
                    # Number of updates per step in environment
                    for i in range(self.config.training.updates_per_step):
                        # Update parameters of all the networks
                        critic_1_loss, critic_2_loss, policy_loss = \
                            self.agent.update_parameters(self.memory, self.config.training.batch_size, updates, self.config.training.delta)
                        updates += 1
                                        
                    self.writer.add_scalar('loss/critic_1', critic_1_loss, updates)
                    self.writer.add_scalar('loss/critic_2', critic_2_loss, updates)
                    self.writer.add_scalar('loss/policy', policy_loss, updates)

                next_state, reward, done, _ = self.env.step(action) # Step        
                episode_steps += 1
                total_numsteps += 1
                episode_reward += np.sum(reward @ w.numpy())

                # Ignore the "done" signal if it comes from hitting the time horizon.

                mask = 1 if episode_steps == self.env._max_episode_steps else float(not done)
                self.memory.push(state, action, w, reward, next_state, mask) # Append transition to memory
                state = next_state

            if total_numsteps > self.config.training.num_steps:
                break

            self.writer.add_scalar('reward/train', episode_reward, total_numsteps)
            logging.info("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, total_numsteps, episode_steps, round(episode_reward, 2)))

            if i_episode % 10 == 0 and self.config.training.eval is True:
                avg_rand_w_reward = 0.
                episodes = 10
                
                for _  in range(episodes):
                    # pick a weight
                    w = self.sampler.sample(1).view(-1)
                    state = self.env.reset()
                    episode_reward = 0
                    done = False
                    while not done:
                        action = self.agent.select_action(state, w, evaluate=True)
                        next_state, reward, done, _ = self.env.step(action)
                        # take inner product
                        reward = (reward * w.numpy()).sum()
                        episode_reward += reward
                        state = next_state
                    avg_rand_w_reward += episode_reward
                avg_rand_w_reward /= episodes
                self.writer.add_scalar('avg_rand_w_reward/test', avg_rand_w_reward, total_numsteps)

                logging.info("----------------------------------------")
                logging.info("Test Episodes: {}, Avg. Reward: {}".format(episodes, round(avg_rand_w_reward, 2)))
                logging.info("----------------------------------------")

            i_episode += 1

            self.total_numsteps = total_numsteps
            self.updates = updates
            self.epi_start = epi_start
            self.i_episode = i_episode
       
        self.env.close()