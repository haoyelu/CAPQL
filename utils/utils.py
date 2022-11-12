import math
import torch
from torch.distributions import Normal

def update_Q(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

class Weight_Sampler_angle:
    def __init__(self, rwd_dim, angle, w = None):
        self.rwd_dim = rwd_dim
        self.angle = angle
        if w is None:
            w = torch.ones(rwd_dim)
        w = w / torch.norm(w)
        self.w = w

    def sample(self, n_sample):

        s = torch.normal(torch.zeros(n_sample, self.rwd_dim))

        # remove fluctuation on dir w
        s = s - (s @ self.w).view(-1, 1) * self.w.view(1, -1)

        # normalize it
        s = s / torch.norm(s, dim = 1, keepdim=True) 

        # sample angle 
        s_angle = torch.rand(n_sample, 1) * self.angle

        # compute shifted vector from w
        w_sample = torch.tan(s_angle) * s + self.w.view(1, -1)

        w_sample = w_sample / torch.norm(w_sample, dim = 1, keepdim=True, p = 1) 

        return w_sample

class Weight_Sampler_pos:
    def __init__(self, rwd_dim):
        self.rwd_dim = rwd_dim

    def sample(self, n_sample):

        # sample from sphrical normal distribution
        s = torch.normal(torch.zeros(n_sample, self.rwd_dim))
        
        # flip all negative weights to be non-negative
        s = torch.abs(s)

        # normalize
        s = s / torch.norm(s, dim = 1, keepdim=True, p = 1) 

        return s