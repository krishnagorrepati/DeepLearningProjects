# Twin-Delayed DDPG

#Complete credit goes to this [awesome Deep Reinforcement Learning 2.0 Course on Udemy](https://www.udemy.com/course/deep-reinforcement-learning/) for the code.

## Importing the libraries

import os
import time
import random
import numpy as np
# import matplotlib.pyplot as plt
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from gym import wrappers
from torch.autograd import Variable
from collections import deque
from PIL import Image as PILImage
from PIL import ImageDraw,ImageOps
import math


class ReplayBuffer(object):

  def __init__(self, max_size=1e6):
    self.storage = []
    self.max_size = max_size
    self.ptr = 0

  def add(self, transition):
    if len(self.storage) == self.max_size:
      self.storage[int(self.ptr)] = transition
      self.ptr = (self.ptr + 1) % self.max_size
    else:
      self.storage.append(transition)

  def sample(self, batch_size):
    ind = np.random.randint(0, len(self.storage), size=batch_size)
    batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones = [], [], [], [], []
    for i in ind: 
      state, next_state, action, reward, done = self.storage[i]
      batch_states.append(np.array(state, copy=False))
      batch_next_states.append(np.array(next_state, copy=False))
      batch_actions.append(np.array(action, copy=False))
      batch_rewards.append(np.array(reward, copy=False))
      batch_dones.append(np.array(done, copy=False))
    return np.array(batch_states), np.array(batch_next_states), np.array(batch_actions), np.array(batch_rewards).reshape(-1, 1), np.array(batch_dones).reshape(-1, 1)


## Step 2: We build one neural network for the Actor model and one neural network for the Actor target
class Actor(nn.Module):
  
  def __init__(self, action_dim, max_action):
    super(Actor, self).__init__()
    self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2)
    self.bn1 = nn.BatchNorm2d(16)
    self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2)
    self.bn2 = nn.BatchNorm2d(32)
    self.conv3 = nn.Conv2d(32,32 ,kernel_size=3, stride=2)
    self.bn3 = nn.BatchNorm2d(32)
    self.conv4 = nn.AvgPool2d(4)
    self.layer_3 = nn.Linear(32, action_dim)
    self.max_action = max_action

  def forward(self, x):
    x = F.relu(self.bn1(self.conv1(x)))
    x = F.relu(self.bn2(self.conv2(x)))
    x = F.relu(self.bn3(self.conv3(x)))
    x = F.relu(self.conv4(x))
    x = x.reshape(1,-1)
    x = self.max_action * torch.tanh(self.layer_3(x))
    return x


class Critic(nn.Module):

  def __init__(self ):
    super(Critic, self).__init__()
    #   Defining the first Critic neural network
    self.features1 = nn.Sequential(
        nn.Conv2d(1 , 16 ,kernel_size = 3 ,stride = 2),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.Conv2d(16 , 32 ,kernel_size = 3 ,stride = 2),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.Conv2d(32 , 32 ,kernel_size = 3 ,stride = 2),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.AvgPool2d(4) 
    )

    self.features2 = nn.Sequential(
        nn.Linear(32+1,300),
        nn.ReLU(),
        nn.Linear(300,400 ),
        nn.ReLU(),
        nn.Linear(400,1)
    )

     # Defining the second Critic neural network
    self.features3 = nn.Sequential(
        nn.Conv2d(1 , 16 ,kernel_size = 3 ,stride = 2),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.Conv2d(16 , 32 ,kernel_size = 3 ,stride = 2),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.Conv2d(32 , 32 ,kernel_size = 3 ,stride = 2),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.AvgPool2d(4) 
    )

    self.features4 = nn.Sequential(
        nn.Linear(32+1,300), # state dim + Action dim
        nn.ReLU(),
        nn.Linear(300,400 ),
        nn.ReLU(),
        nn.Linear(400,1)
    )

  def forward(self, x, y):

    # Forward-Propagation on the first Critic Neural Network

    x1 = self.features1(x)
    x1 = x1.view(x1.size(0),-1)
    y1 = y.view(y.size(0),-1)
    merged_1 = torch.cat((x1,y1),dim=1)
    merged_1 = self.features2(merged_1)

    # Forward-Propagation on the second Critic Neural Network
    x2 = self.features3(x)
    x2 = x2.view(x2.size(0),-1)
    y2 = y.view(y.size(0),-1)
    merged_2 = torch.cat((x2,y2),dim=1)
    merged_2 = self.features4(merged_2)
    return merged_1 , merged_2

  def Q1(self, x, y):
    x1 = self.features1(x)
    x1 = x1.view(x1.size(0),-1)
    y1 = y.view(y.size(0),-1)
    merged_1 = torch.cat((x1,y1),dim=1)
    merged_1 = self.features2(merged_1)
    return merged_1

  
## Steps 4 to 15: Training Process

# Building the whole Training Process into a class

class TD3(object):
  
  def __init__(self, action_dim, max_action):
    self.actor = Actor( action_dim, max_action)
    self.actor_target = Actor( action_dim, max_action)
    self.actor_target.load_state_dict(self.actor.state_dict())
    self.actor_optimizer = torch.optim.Adam(self.actor.parameters())

    self.critic = Critic()
    self.critic_target = Critic()
    self.critic_target.load_state_dict(self.critic.state_dict())
    self.critic_optimizer = torch.optim.Adam(self.critic.parameters())
    self.max_action = max_action

  def select_action(self, state):
    state = torch.Tensor(state).unsqueeze(0).unsqueeze(0)
    return self.actor(state).cpu().data.numpy().flatten()

  def train(self, replay_buffer, iterations, batch_size=100, discount=0.99, tau=0.005, policy_noise=0.2, noise_clip=0.5,
    policy_freq=2):
    
    for it in range(iterations):
      
      # Step 4: We sample a batch of transitions (s, s’, a, r) from the memory
      batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones = replay_buffer.sample(batch_size)
      state = torch.Tensor(batch_states)
      next_state = torch.Tensor(batch_next_states)
      action = torch.Tensor(batch_actions)
      reward = torch.Tensor(batch_rewards)
      done = torch.Tensor(batch_dones)
      
      # Step 5: From the next state s’, the Actor target plays the next action a’
      next_action = self.actor_target(next_state)
      
      # Step 6: We add Gaussian noise to this next action a’ and we clamp it in a range of values supported by the environment
      noise = torch.Tensor(batch_actions).data.normal_(0, policy_noise)
      noise = noise.clamp(-noise_clip, noise_clip)
      next_action = (next_action + noise).clamp(-self.max_action, self.max_action)
      
      # Step 7: The two Critic targets take each the couple (s’, a’) as input and return two Q-values Qt1(s’,a’) and Qt2(s’,a’) as outputs
      target_Q1, target_Q2 = self.critic_target(next_state, next_action)
      
      # Step 8: We keep the minimum of these two Q-values: min(Qt1, Qt2)
      target_Q = torch.min(target_Q1, target_Q2)
      
      # Step 9: We get the final target of the two Critic models, which is: Qt = r + γ * min(Qt1, Qt2), where γ is the discount factor
      target_Q = reward + ((1 - done) * discount * target_Q).detach()
      
      # Step 10: The two Critic models take each the couple (s, a) as input and return two Q-values Q1(s,a) and Q2(s,a) as outputs
      current_Q1, current_Q2 = self.critic(state, action)
      
      # Step 11: We compute the loss coming from the two Critic models: Critic Loss = MSE_Loss(Q1(s,a), Qt) + MSE_Loss(Q2(s,a), Qt)
      critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
      
      # Step 12: We backpropagate this Critic loss and update the parameters of the two Critic models with a SGD optimizer
      self.critic_optimizer.zero_grad()
      critic_loss.backward()
      self.critic_optimizer.step()
      
      # Step 13: Once every two iterations, we update our Actor model by performing gradient ascent on the output of the first Critic model
      if it % policy_freq == 0:
        actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Step 14: Still once every two iterations, we update the weights of the Actor target by polyak averaging
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
          target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        
        # Step 15: Still once every two iterations, we update the weights of the Critic target by polyak averaging
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
          target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    # Making a save method to save a trained model
  def save(self, filename, directory):
    torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename))
    torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, filename))
  
  # Making a load method to load a pre-trained model
  def load(self, filename, directory):
    self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename)))
    self.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, filename)))

