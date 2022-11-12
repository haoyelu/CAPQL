import sys
# sys.modules[__name__]
from gym.envs.mujoco.hopper_v3 import HopperEnv
from gym.envs.mujoco.walker2d_v3 import Walker2dEnv
from gym.envs.mujoco.half_cheetah_v3 import HalfCheetahEnv
from gym.envs.mujoco.ant_v3  import AntEnv
from gym.envs.mujoco.humanoid_v3 import HumanoidEnv
from gym.envs.mujoco.swimmer_v3 import SwimmerEnv
import numpy as np
import gym

def register_env():
    try:
        gym.envs.register(
            id='HopperM-v0',
            entry_point='envs.env:HopperEnv_m',
            max_episode_steps=1000,
        )
    except Exception as err:
        print(f"Unexpected {err=}, {type(err)=}")
    
    try:
        gym.envs.register(
            id='Walker2dM-v0',
            entry_point='envs.env:Walker2dEnv_m',
            max_episode_steps=1000,
        )
    except Exception as err:
        print(f"Unexpected {err=}, {type(err)=}")

    try:
        gym.envs.register(
            id='HalfCheetahM-v0',
            entry_point='envs.env:HalfCheetahEnv_m',
            max_episode_steps=1000,
            reward_threshold=4800.0,
        )
    except Exception as err:
        print(f"Unexpected {err=}, {type(err)=}")

   
    try:
        gym.envs.register(
            id='HumanoidEnvM-v0',
            entry_point='envs.env:HumanoidEnv_m',
            max_episode_steps=1000,
        )
    except Exception as err:
        print(f"Unexpected {err=}, {type(err)=}")




class HopperEnv_m(HopperEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.rwd_dim = 3


    def step(self, action):
        x_position_before = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        x_position_after = self.sim.data.qpos[0]
        x_velocity = (x_position_after - x_position_before) / self.dt

        # control cost is positive
        ctrl_cost = self.control_cost(action)
        
        forward_reward = self._forward_reward_weight * x_velocity
        healthy_reward = self.healthy_reward

        rewards = forward_reward + healthy_reward
        costs = ctrl_cost

        observation = self._get_obs()

        done = self.done
        info = {
            "x_position": x_position_after,
            "x_velocity": x_velocity,
        }

        reward = np.array([forward_reward, healthy_reward, -costs])
        
        return observation, reward, done, info

class Walker2dEnv_m(Walker2dEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.rwd_dim = 3

    def step(self, action):
        x_position_before = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        x_position_after = self.sim.data.qpos[0]
        x_velocity = (x_position_after - x_position_before) / self.dt

        # control cost is positive
        ctrl_cost = self.control_cost(action)
        forward_reward = self._forward_reward_weight * x_velocity
        healthy_reward = self.healthy_reward

        rewards = forward_reward + healthy_reward
        costs = ctrl_cost

        observation = self._get_obs()

        done = self.done
        info = {
            "x_position": x_position_after,
            "x_velocity": x_velocity,
        }

        reward = np.array([forward_reward, healthy_reward, -costs])
        
        return observation, reward, done, info


class HalfCheetahEnv_m(HalfCheetahEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.rwd_dim = 2

    def step(self, action):
        x_position_before = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        x_position_after = self.sim.data.qpos[0]
        x_velocity = (x_position_after - x_position_before) / self.dt
        # control cost is positive
        ctrl_cost = self.control_cost(action)
        forward_reward = self._forward_reward_weight * x_velocity
        observation = self._get_obs()


        observation = self._get_obs()
        done = False

        info = {
            "x_position": x_position_after,
            "x_velocity": x_velocity,
            "reward_run": forward_reward,
            "reward_ctrl": -ctrl_cost,
        }

        reward = np.array([forward_reward, -ctrl_cost])
        
        return observation, reward, done, info

class HumanoidEnv_m(HumanoidEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.rwd_dim = 4

    def step(self, action):
        xy_position_before = mass_center(self.model, self.sim)
        self.do_simulation(action, self.frame_skip)
        xy_position_after = mass_center(self.model, self.sim)

        xy_velocity = (xy_position_after - xy_position_before) / self.dt
        x_velocity, y_velocity = xy_velocity

        ctrl_cost = self.control_cost(action)
        contact_cost = self.contact_cost

        forward_reward = self._forward_reward_weight * x_velocity
        healthy_reward = self.healthy_reward

        rewards = forward_reward + healthy_reward
        costs = ctrl_cost + contact_cost

        observation = self._get_obs()
        reward = rewards - costs
        done = self.done
        info = {
            "reward_linvel": forward_reward,
            "reward_quadctrl": -ctrl_cost,
            "reward_alive": healthy_reward,
            "reward_impact": -contact_cost,
            "x_position": xy_position_after[0],
            "y_position": xy_position_after[1],
            "distance_from_origin": np.linalg.norm(xy_position_after, ord=2),
            "x_velocity": x_velocity,
            "y_velocity": y_velocity,
            "forward_reward": forward_reward,
        }

        reward = np.array([forward_reward, healthy_reward, -ctrl_cost, -contact_cost])

        return observation, reward, done, info

def mass_center(model, sim):
    mass = np.expand_dims(model.body_mass, axis=1)
    xpos = sim.data.xipos
    return (np.sum(mass * xpos, axis=0) / np.sum(mass))[0:2].copy()