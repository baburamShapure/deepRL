import gym 
import torch 
import torch.nn as nn 
import numpy as np 
import math 
import random  
import torch.optim as optim 
import torch.nn.functional as F 


class dqn(nn.Module):
    
    def __init__(self):
        super(dqn, self).__init__()
        self.fc1 = nn.Linear(4, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, 2)
    
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x 


policy_agent =  dqn()
target_agent = dqn() 

optimizer = optim.Adam(policy_agent.parameters())
REPLAY = [] 
batch_size = 32
GAMMA = 0.99 
update_count = 0 

env = gym.make('CartPole-v0')

for episode in range(200):
    state = env.reset()
    env.render()
    episode_reward  = 0
    for t in range(500):
        if np.random.uniform() < 0.2:
            action = random.sample([0, 1], 1)[0]
        else: 
            t_state = torch.FloatTensor(state)
            with torch.no_grad():
                action = policy_agent(t_state).argmax().item()
        
        next_state, reward, done, _ = env.step(action)
        episode_reward += reward
        REPLAY.append((state, action, reward, next_state, done))
        state = next_state

        if len(REPLAY) > batch_size: 
            # gradient descent. 
            minibatch = random.sample(REPLAY, batch_size)

            states = [x[0] for x in minibatch]
            actions = [x[1] for x in minibatch]
            rewards = [x[2] for x in minibatch]
            next_states = [x[3] for x in minibatch]
            dones = [x[4] for x in minibatch] 

            states = torch.FloatTensor(states)
            actions = torch.LongTensor(actions)
            rewards = torch.FloatTensor(rewards)
            next_states = torch.FloatTensor(next_states)
            dones = torch.FloatTensor(dones)

            with torch.no_grad():
                q_targets = rewards + (1 - dones) * GAMMA * target_agent(next_states).max(1)[0]
            
            q_response = policy_agent(states).gather(1, actions.unsqueeze(1)).squeeze()

            loss = F.mse_loss(q_targets, q_response)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            update_count += 1

            if update_count % 100 == 0:
                target_agent.load_state_dict(policy_agent.state_dict())

        if done: 
            print('Episode: {}   Reward: {}'.format(episode + 1, episode_reward))
            break 
