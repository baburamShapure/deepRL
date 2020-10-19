"""
vanilla Actor critic 
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
    def __init__(self, input_dim, hidden_dim, n_actions): 
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
MAX_EPISODES = 500
MAX_STEPS_PER_EPISODE = 200
GAMMA= 0.99
LR = 0.01
env = gym.make('CartPole-v0')
OBS_SPACE_DIM = env.observation_space.shape[0]
ACTION_DIM = env.action_space.n
HIDDEN_DIM = 128
policynet = PolicyNet(OBS_SPACE_DIM, HIDDEN_DIM, ACTION_DIM )
valuenet = ValueNet(OBS_SPACE_DIM, HIDDEN_DIM, ACTION_DIM)
valuenet_ = copy.deepcopy(valuenet)
policyoptimizer = optim.Adam(policynet.parameters())
valueoptimizer = optim.Adam(valuenet.parameters())

REWARDS =[] 


for episode in tqdm.tqdm(range(MAX_EPISODES)): 
    ep_reward = 0 
    state = env.reset()
    state = torch.tensor(state).float()
    action_probs = policynet(state)
    #choose an action at random: 
    action = distributions.Categorical(action_probs).sample().item()
    q_val = valuenet(state)[action]

    for t in range(MAX_STEPS_PER_EPISODE): 
        s_, r, done, __ = env.step(action)        

        # actor loss: 
        actor_loss = - q_val.item() * torch.log(action_probs[action])

        policyoptimizer.zero_grad()
        actor_loss.backward()
        policyoptimizer.step()

        #choose next action but don't execute.  
        s_ = torch.tensor(s_).float()
        next_action_probs = policynet(s_)
        next_action = distributions.Categorical(next_action_probs).sample().item()
        next_q_val = valuenet_(s_)[next_action]

        # critic
    
        critic_loss = F.mse_loss(torch.tensor(r + (~done) * GAMMA * next_q_val.item()),  q_val )
        valueoptimizer.zero_grad()
        critic_loss.backward()
        valueoptimizer.step()

        state = s_
        action = next_action
        action_probs = policynet(state)
        q_val = valuenet(state)[action]

        ep_reward += r 

        if done:
            # print("episode {0} - Reward {1}".format(episode, ep_reward))
            break
    
    if episode % 50: 
        valuenet_.load_state_dict(valuenet.state_dict())

    REWARDS.append(ep_reward)
    # if episode % 5 == 0: 
    #     print("episode: {0}, mean_reward {1}".format(episode, np.mean(REWARDS)))
    
    
plt.plot(REWARDS)
plt.show()