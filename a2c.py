"""
advantage actor critic
"""

import torch 
import torch.nn as nn 
import torch.optim as optim 
import torch.nn.functional as F 
import torch.distributions as distributions
import gym
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import tqdm 
import copy


class PolicyNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_actions): 
        super(PolicyNet, self).__init__()
        self.input_dim = input_dim 
        self.hidden_dim = hidden_dim 
        self.n_actions = n_actions 
        self.linear1 = nn.Linear(self.input_dim, 
                                self.hidden_dim)
        self.policy_head = nn.Linear(self.hidden_dim, 
                                self.n_actions)
    def forward(self, x): 
        x = F.relu(self.linear1(x))
        # policy:
        policy = F.softmax(self.policy_head(x), dim = 0)
        return policy

class ValueNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_actions= 1): 
        super(ValueNet, self).__init__()
        self.input_dim = input_dim 
        self.hidden_dim = hidden_dim 
        self.n_actions = n_actions 
        self.linear1 = nn.Linear(self.input_dim, 
                                self.hidden_dim)
        self.value_head = nn.Linear(self.hidden_dim, 
                                self.n_actions)
    def forward(self, x): 
        x = F.relu(self.linear1(x))
        # policy:
        value = self.value_head(x)
        return value


torch.autograd.set_detect_anomaly(True)
MAX_EPISODES = 5000
MAX_STEPS_PER_EPISODE = 200
GAMMA= 0.99
LR = 0.01
env = gym.make('CartPole-v0')
OBS_SPACE_DIM = env.observation_space.shape[0]
ACTION_DIM = env.action_space.n
HIDDEN_DIM = 128
policynet = PolicyNet(OBS_SPACE_DIM, HIDDEN_DIM, ACTION_DIM )
valuenet = ValueNet(OBS_SPACE_DIM, HIDDEN_DIM)
valuenet_ = copy.deepcopy(valuenet)
policyoptimizer = optim.Adam(policynet.parameters())
valueoptimizer = optim.Adam(valuenet.parameters())

REWARDS =[] 


for episode in tqdm.tqdm(range(MAX_EPISODES)):
    state = env.reset() 
    ep_reward = 0 
    for t in range(MAX_STEPS_PER_EPISODE): 

        state = torch.tensor(state).float()

        #get current policy.
        action_probs = policynet(state)

        # sample action from current policy. 
        action = distributions.Categorical(action_probs).sample().item()

        # execute action and get new state & reward. 
        state_, r, done, __ = env.step(action)

        state_ = torch.tensor(state_).float()

        # get advantage. 
        adv = r + (~done) * GAMMA * valuenet(state_).item() - valuenet(state)

        # update actor. 
        actor_loss = -torch.log(action_probs[action]) * adv.item()
        policyoptimizer.zero_grad()
        actor_loss.backward()
        policyoptimizer.step()

        # update critic. 
        critic_loss = pow(adv, 2)
        valueoptimizer.zero_grad() 
        critic_loss.backward()
        valueoptimizer.step()

        ep_reward += r 

        if done:
            break 
    
    REWARDS.append(ep_reward)

plt.plot(REWARDS)
plt.show()