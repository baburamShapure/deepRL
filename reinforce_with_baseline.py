import numpy as np 
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim 
import torch.distributions as distributions 
import gym 
import matplotlib.pyplot as plt 
import pandas as pd 

"""
Let’s see how it works in a simple action-value actor-critic algorithm.

Initialize s,θ,w

at random; sample a∼πθ(a|s)
.
For t=1…T
:
Sample reward rt∼R(s,a)

and next state s′∼P(s′|s,a)
;
Then sample the next action a′∼πθ(a′|s′)
;
Update the policy parameters: θ←θ+αθQw(s,a)∇θlnπθ(a|s)
;
Compute the correction (TD error) for action-value at time t:
δt=rt+γQw(s′,a′)−Qw(s,a)

and use it to update the parameters of action-value function:
w←w+αwδt∇wQw(s,a)
Update a←a′
and s←s′.
"""

class PolicyNet(nn.Module): 
    def __init__(self, INPUT_DIM, HIDDEN_DIM, OUT_DIM): 
        super(PolicyNet, self).__init__()
        self.linear1 = nn.Linear(INPUT_DIM, HIDDEN_DIM)
        self.linear2 = nn.Linear(HIDDEN_DIM, OUT_DIM)

    def forward(self, x): 
        out = F.relu(self.linear1(x))
        out = F.softmax(self.linear2(out), dim = 0 )
        return out

class ValueEstimator(nn.Module): 
    def __init__(self, INPUT_DIM, HIDDEN_DIM): 
        super(ValueEstimator, self).__init__()
        self.linear1 = nn.Linear(INPUT_DIM, HIDDEN_DIM)
        self.linear2 = nn.Linear(HIDDEN_DIM, 1)
    
    def forward(self, x): 
        out = F.relu(self.linear1(x))
        out = self.linear2(out)
        return out 



GAMMA = 0.99
env = gym.make('CartPole-v0')
inputdim = env.observation_space.shape[0]
outputdim = env.action_space.n
hiddendim = 128


policy = PolicyNet(inputdim, hiddendim, outputdim)
value_estimator = ValueEstimator(inputdim, hiddendim)
optimizer = optim.Adam(policy.parameters())

critic_optimizer = optim.Adam(value_estimator.parameters())

MAX_EPISODES = 1000
STEPS_EPISODE = 150

EP_REWARD = []
for episode in range(MAX_EPISODES): 
    state = env.reset()
    ACTIONS = []
    LOGPROBS = [] 
    REWARDS = []
    VALUES = []
    # play out 1 entire episode till termination or MAX_STEPS.
    for step in range(STEPS_EPISODE): 
        state = torch.tensor(state).float()
        # how good is this state ? the baseline. (& later the critic. remember to bootstrap)
        value = value_estimator(state)
        # what is the policy given this state? 
        action_probs = policy(state)
        action = distributions.Categorical(action_probs).sample().item()
        # take this action. 
        next_state, reward, done, __ = env.step(action)
        
        state = next_state
        # keep track of values, actions, logprobs & rewards. 
        ACTIONS.append(action)
        VALUES.append(value)
        LOGPROBS.append(torch.log(action_probs[action]))
        REWARDS.append(reward)
        if done: 
            break  
    
    # episode terminated. now perform updates. 
    # compute discounted reward for each step. G_t for each t. 
    EP_REWARD.append(sum(REWARDS))
    DISCOUNTED_REWARDS = [REWARDS[-1]]
    gamma_pow = 1
    for r in REWARDS[::-1][1:]: 
        DISCOUNTED_REWARDS.insert(0, r + (GAMMA * DISCOUNTED_REWARDS[0]))
    
    policy_grads = []
    for g_t, logprobs, b_t in zip(DISCOUNTED_REWARDS, LOGPROBS, VALUES): 
        #subtract baseline from g_t .
        policy_grads.append(- logprobs * (g_t - b_t.item()))
    policy_grads = torch.stack(policy_grads)

    # update policy net. 
    loss = policy_grads.sum()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()    

    # refit baseline. 
    VALUES = torch.stack(VALUES)
    DISCOUNTED_REWARDS = torch.tensor(DISCOUNTED_REWARDS)
    critic_loss = F.mse_loss(VALUES, DISCOUNTED_REWARDS)
    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()

    print('Episode: {0}, Reward: {1}'.format(episode, sum(REWARDS))) 

plt.plot(EP_REWARD)
plt.show()

pd.DataFrame(EP_REWARD, columns = ['reward']).to_clipboard()