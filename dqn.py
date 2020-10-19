import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim 
import gym 
import numpy as np 
import random
from matplotlib import pyplot as plt 
import tqdm
import copy

# class ReplayMemory():
#     def __init__(self): 

def sample_minibatch(replay, batch_size): 
    """sample a minibatch from a replay memory. 
    replay -> (s, a, r, s', done)
    """
    minibatch = random.sample(replay, batch_size)
    s = torch.tensor([i[0] for i in minibatch]).float()
    a = torch.tensor([i[1] for i in minibatch])
    r = torch.tensor([i[2] for i in minibatch])
    s_ = torch.tensor([i[3] for i in minibatch]).float()
    done = torch.tensor([i[4] for i in minibatch])
    return s, a, r, s_, done



class DQNAgent(nn.Module):
    def __init__(self, input_dim, hidden_dim, nactions): 
        super(DQNAgent, self).__init__()        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.nactions = nactions 
        self.linear1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.linear2 = nn.Linear(self.hidden_dim, self.nactions)
    def forward(self, x): 
        out = F.relu(self.linear1(x))
        out = self.linear2(out)
        return out 

# constants. 
GAMMA = 0.99 
LR = 0.001
HIDDEN_DIM = 128
MAX_EPISODES = 10000
STEPS_PER_EPISODE = 300 
GREEDY_EPSILON = 0.2
MINIBATCH = 128

env = gym.make('CartPole-v0')        
observation_dim = env.observation_space.shape[0]
n_actions = env.action_space.n 

valueEstimator = DQNAgent(observation_dim, HIDDEN_DIM, n_actions)
valueEstimator_ = copy.deepcopy(valueEstimator)

optimizer = optim.Adam(valueEstimator.parameters())

state = env.reset()
state_tensor= torch.tensor(state).float()
values = valueEstimator(state_tensor)

TOTAL_REWARDS = []
REPLAY = []
for episode in tqdm.tqdm(range(MAX_EPISODES)): 
    ep_reward = 0
    # reset environment at start of episode & get new state. 
    state = env.reset()
    for t in range(STEPS_PER_EPISODE):
        
        with torch.no_grad():
            state_tensor = torch.tensor(state).float()
            q_estimates = valueEstimator(state_tensor)
        # epsilon-greedy. 
        if random.uniform(0, 1) > GREEDY_EPSILON: 
            # happens with prob 1 - GREEDY_EPSILON
            # greedy action. 
            action = torch.argmax(q_estimates).item()
        else: 
            # choose an action at random. 
            action = env.action_space.sample()

        # execute the action & get new state. 
        new_state, reward, done, _ = env.step(action)
        ep_reward += reward
        # append transition to replay memory. 
        REPLAY.append((state, action, reward, new_state, done))
        # get ready for new step. 
        state = new_state 
        if done: 
            break 
    TOTAL_REWARDS.append(ep_reward)
    # at end of episode, sample a minibatch from REPLAY memory. 
    if len(REPLAY) > MINIBATCH:
        s, a, r, s_, dones = sample_minibatch(REPLAY, MINIBATCH)
        # compute the targets. 
        with torch.no_grad():
            q_targets = r +  (~dones) * GAMMA  * torch.max(valueEstimator_(s_), 1)[0]

        # compute the estimates. 
        q_estimates = torch.gather(valueEstimator(s), 1, a.view(-1, 1))

        loss = F.mse_loss(q_estimates.squeeze(), q_targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if episode % 100 == 0: 
        valueEstimator_.load_state_dict(valueEstimator.state_dict())


plt.plot(TOTAL_REWARDS)
plt.show()
