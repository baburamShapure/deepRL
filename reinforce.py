"""
REINFORCE: 

Repeat until convergence: 
    1. Roll out trajectory according to current policy. 
    2. Compute log probabilities and reward at each step. 
    3. Compute discounted future reward at each  step 
    4. Compute policy gradient and update parameter. 

    Inspiration: https://medium.com/@thechrisyoon/deriving-policy-gradients-and-implementing-reinforce-f887949bd63
"""

import gym 
import numpy as np 
import torch 
import torch.nn as nn 
from torch import optim 
import torch.nn.functional as F 
from torch import  distributions
import matplotlib.pyplot as plt 

class PolicyNetwork(nn.Module): 
    def __init__(self):
        super(PolicyNetwork, self).__init__() 
        self.n_actions = 2
        self.input_dim = 4
        self.hidden_dim = 128
        self.linear1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.linear2 = nn.Linear( self.hidden_dim, self.n_actions )
    def forward(self, x): 
        out = F.relu(self.linear1(x))
        out = self.linear2(out)
        out = F.softmax(out, dim = 0)        
        return out 

policynet = PolicyNetwork()
# policynet(torch.tensor(state).float())

optimizer = optim.Adam(policynet.parameters())
# roll out a trajectory using this policy. 
env = gym.make('CartPole-v0')
state = env.reset()

MAX_EPISODES = 500
MAX_STEP = 100
GAMMA = 0.99

EP_REWARDS = [] 
for episode in range(MAX_EPISODES): 
    R = []
    LOGPROBS = [] 
    state = env.reset()
    for each_step in range(MAX_STEP): 
        # play out a trajectory using current policy. 
        state = torch.tensor(state).float()
        actiondist = policynet(state)    
        action = distributions.Categorical(actiondist).sample()
        LOGPROBS.append(torch.log(actiondist[action]))
        state, reward, done, __ = env.step(action.item())
        R.append(reward)
        if done: 
            break 
    print('episode# {0}, steps: {1}'.format(episode, each_step))    
    EP_REWARDS.append(each_step)
    
    # now update.
    DISC_REWARDS = []  
    for t in range(each_step + 1): 
        # compute Gt. 
        rt = R[t: ]
        Gt = 0
        for i in range(len(rt)): 
            Gt += (GAMMA ** i) * rt[i]
        DISC_REWARDS.append(Gt)

    policy_grads = []
    for i,j in zip(DISC_REWARDS, LOGPROBS): 
        policy_grads.append(i * (- j))  # because pytorch only minimizes. 
    policy_grads = torch.stack(policy_grads).sum()

    optimizer.zero_grad()
    policy_grads.backward()
    optimizer.step()

plt.plot(EP_REWARDS)
plt.show()

