import os
import torch
import copy
import torch.nn.functional as F
from torch.optim import Adam
from utils.utils import update_Q
from .networks import GaussianPolicy, QNetwork

class CAPQL(object):
    def __init__(self, num_inputs, action_space, rwd_dim, config):

        self.gamma = config.training.gamma
        self.tau = config.training.tau
        self.alpha = config.training.alpha

        self.policy_type = config.model.type
        self.target_update_interval = config.training.target_update_interval
        self.device = torch.device("cuda" if config.gpu.cuda else "cpu")

        # define TWO Q networks for training
        self.critic = QNetwork(num_inputs, action_space.shape[0], rwd_dim, config.model.hidden_size).to(device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=config.training.lr)

        # define TWO Q networks for target
        self.critic_target =  copy.deepcopy(self.critic)
        self.policy = GaussianPolicy(num_inputs, action_space.shape[0], rwd_dim, config.model.hidden_size, action_space).to(self.device)
        self.policy_optim = Adam(self.policy.parameters(), lr=config.training.lr)

    def select_action(self, state, w, evaluate=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        w = torch.FloatTensor(w).to(self.device).unsqueeze(0)

        if evaluate is False:
            action, _, _ = self.policy.sample(state, w)
        else:
            _, _, action = self.policy.sample(state, w)

        return action.detach().cpu().numpy()[0]
    
    def update_parameters(self, memory, batch_size, updates, delta = 0.1):
        ## Sample a batch from memory
        state_batch, action_batch, w_batch, reward_batch, next_state_batch, mask_batch = memory.sample(batch_size=batch_size)
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        w_batch = torch.FloatTensor(w_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)

        ##########################
        ## train the Q network  ##
        ##########################

        # compute next_q_value target
        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch, w_batch)
            
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action, w_batch, h_op=False)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi 

            next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)

        # update
        qf1, qf2 = self.critic(state_batch, action_batch, w_batch)
        qf1_loss = F.mse_loss(qf1, next_q_value)  
        qf2_loss = F.mse_loss(qf2, next_q_value)  
        qf_loss = qf1_loss + qf2_loss

        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        ##############################
        ## train the policy network ##
        ##############################

        # for each sample in minibatch, sample an action in 3-d space
        # pi is the action, log_pi is its log probability
        pi, log_pi, _ = self.policy.sample(state_batch, w_batch) 
        qf1_pi, qf2_pi = self.critic(state_batch, pi, w_batch, h_op = False)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        # minimize KL divergence
        min_qf_pi = (min_qf_pi * w_batch).sum(dim=-1, keepdim = True)       
        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean() 

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        # sync the Q networks
        if updates % self.target_update_interval == 0:
            update_Q(self.critic_target, self.critic, self.tau)

        return qf1_loss.item(), qf2_loss.item(), policy_loss.item()