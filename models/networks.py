import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import copy

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

class ValueNetwork(nn.Module):
    def __init__(self, num_inputs, hidden_dim):
        super(ValueNetwork, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x

class QNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, rwd_dim, hidden_dim):
        super(QNetwork, self).__init__()

        # Q1 architecture
        self.Q1 = nn.Sequential(
            nn.Linear(num_inputs + num_actions + rwd_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, rwd_dim)
        )

        # Q2 architecture
        self.Q2 = copy.deepcopy(self.Q1)
        self.rwd_dim = rwd_dim
        self.apply(weights_init_)

    def forward(self, state, action, w, h_op=False):
        if not h_op:
            # 1) Prepare input
            # xu: batch_size x (state_dim + action_dim + rwd_dim)            
            xu = torch.cat([state, action, w], 1)

            # 2) Evaluate 
            # x1, x2: batch_size x rwd_dim
            x1 = self.Q1(xu)
            x2 = self.Q2(xu)
            return x1, x2
        else:
            
            ## For qenv_ctn H operation

            with torch.no_grad():
                batch_size = n_weight = len(state)

                # 1) Prepare input
                # xu: batch_size x (state_dim + action_dim)            
                state_action = torch.cat([state, action], 1)

                # state_action: batch_size x (state_dim + action_dim) -> batch_size x 1 x (state_dim + action_dim)
                # w: n_prob_weight x rwd_dim -> 1 x n_prob_weight x rwd_dim
                # Concatenate to get: batch_size x n_prob_weight x (state_dim + action_dim + rwd_dim)

                xu_expand = torch.cat(
                    (
                        state_action.unsqueeze(1).expand(-1, n_weight, -1),
                        w.unsqueeze(0).expand(batch_size, -1, -1),
                    ),
                    dim = -1
                )

                # 2) Evaluate to get Q values
                # q1_expand, q2_expand: batch_size x n_prob_weight x rwd_dim
                q1_expand = self.Q1(xu_expand)
                q2_expand = self.Q2(xu_expand)
                q_expand = torch.stack([q1_expand, q2_expand], 2).view(batch_size, n_weight * 2, self.rwd_dim)

                # 3) Compute projection
                # w: batch_size x rwd_dim
                # q1_expand, q2_expand: batch_size x n_prob_weight x rwd_dim
                # proj_1, proj_2: batch_size x n_prob_weight
                proj_1 = (w.unsqueeze(1) * q1_expand).sum(-1)
                proj_2 = (w.unsqueeze(1) * q2_expand).sum(-1)

                # max_proj_1, max_proj_2: batch_size
                # max_id_1, max_id_2: batch_size
                max_proj_1, max_id_1 = torch.max(proj_1, dim = 1)
                max_proj_2, max_id_2 = torch.max(proj_2, dim = 1)

                # find the network gives the smaller projection
                # first_net_smaller_mask: batch_size
                first_net_smaller_mask = (max_proj_1 < max_proj_2).int().unsqueeze(-1)


            # compute again for the max the projection with gradient recorded
            q1_max = self.Q1(torch.cat([state, action, w[max_id_1]], 1))
            q2_max = self.Q2(torch.cat([state, action, w[max_id_2]], 1))
            q = q1_max * first_net_smaller_mask + q2_max * (1 - first_net_smaller_mask)
    
            return q

class GaussianPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, rwd_dim, hidden_dim, action_space=None):
        super(GaussianPolicy, self).__init__()
        
        self.linear1 = nn.Linear(num_inputs + rwd_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state, w):
        
        state_comp = torch.cat((state, w), dim = 1)

        x = F.relu(self.linear1(state_comp))
        x = F.relu(self.linear2(x))

        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)

        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)

        return mean, log_std


    def sample(self, state, w):
        # for each state in the mini-batch, get its mean and std
        mean, log_std = self.forward(state, w)
        std = log_std.exp()

        # sample actions
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))

        # restrict the outputs
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias

        # compute the prob density of the samples
        
        log_prob = normal.log_prob(x_t)

        # Enforcing Action Bound
        # compute the log_prob as the normal distribution sample is processed by tanh 
        #       (reparameterization trick)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        log_prob = log_prob.clamp(-1e3, 1e3)       

        mean = torch.tanh(mean) * self.action_scale + self.action_bias

        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device)