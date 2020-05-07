# Twin-Delayed DDPG

#Complete credit goes to this [awesome Deep Reinforcement Learning 2.0 Course on Udemy](https://www.udemy.com/course/deep-reinforcement-learning/) for the code.

## Importing the libraries

import os
import time
import random
import numpy as np
import matplotlib.pyplot as plt
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

global cr_loss
global ac_loss
global c_iterns
global a_iterns
cr_loss =[]
ac_loss = []

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
    batch_states_0,batch_states_1, batch_next_states_0,batch_next_states_1, batch_actions, batch_rewards, batch_dones = [],[], [], [], [], [], []
    for i in ind: 
      state, next_state, action, reward, done = self.storage[i]    
      batch_states_0.append(np.array(state[0], copy=False))
      batch_states_1.append(np.array(state[1], copy=False))
      batch_next_states_0.append(np.array(next_state[0], copy=False))
      batch_next_states_1.append(np.array(next_state[1], copy=False))      
      batch_actions.append(np.array(action, copy=False))
      batch_rewards.append(np.array(reward, copy=False))
      batch_dones.append(np.array(done, copy=False))
    return np.array(batch_states_0),np.array(batch_states_1), np.array(batch_next_states_0),np.array(batch_next_states_1), np.array(batch_actions), np.array(batch_rewards).reshape(-1, 1), np.array(batch_dones).reshape(-1, 1)


## Step 2: We build one neural network for the Actor model and one neural network for the Actor target
class Actor(nn.Module):
  
  def __init__(self, state_dim,action_dim, max_action):
    super(Actor, self).__init__()

    self.actor_2d = nn.Sequential(
        nn.Conv2d(1 , 16 ,kernel_size = 3 ,stride = 2),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.Conv2d(16 , 32 ,kernel_size = 3 ,stride = 2),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.Conv2d(32 , 64 ,kernel_size = 3 ,stride = 2),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.AvgPool2d(4)
        # nn.BatchNorm1d(64)                    # check if needed
    )

    self.actor_1d = nn.Sequential(
        nn.Linear(state_dim,50),
        # nn.BatchNorm1d(50),
        nn.ReLU(),
        nn.Linear(50,100)
        # nn.BatchNorm1d(100)                 # check if needed
    )

    self.actor_merged = nn.Sequential(
        nn.Linear(100+64 , 200), # Combining feature vectors
        # nn.BatchNorm1d(200),
        nn.ReLU(),
        nn.Linear(200,250),
        # nn.BatchNorm1d(250),
        nn.ReLU(),
        nn.Linear(250,75),
        nn.ReLU(),
        nn.Linear(75,action_dim)
    )

    self.max_action = max_action

  def forward(self, state_img,state_pos):

    x = self.actor_2d(state_img)
    x = x.view(x.size(0),-1)

    y =  self.actor_1d(state_pos)
    y = y.view(y.size(0),-1)

    merged = torch.cat((x,y),dim=1)
    merged = self.actor_merged(merged)
  
    merged = torch.Tensor(self.max_action) * torch.tanh(merged)
    return merged


class Critic(nn.Module):

  def __init__(self , state_dim,action_dim ):
    super(Critic, self).__init__()
    #   Defining the first Critic neural network
    self.features1 = nn.Sequential(
        nn.Conv2d(1 , 16 ,kernel_size = 3 ,stride = 2),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.Conv2d(16 , 32 ,kernel_size = 3 ,stride = 2),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.Conv2d(32 , 64 ,kernel_size = 3 ,stride = 2),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(output_size=(1,1))
    )

    self.linear_1 = nn.Linear(state_dim + action_dim,20)
    # self.bn_c1 = nn.BatchNorm1d(20)
    self.linear_11 = nn.Linear(20,50)
    # self.bn_c11 = nn.BatchNorm1d(50)
    self.linear_12 = nn.Linear(50,100)
    # self.bn_c12 = nn.BatchNorm1d(100)

    self.features2 = nn.Sequential(
        nn.Linear(64 + 100 ,200),
        # nn.BatchNorm1d(200),
        nn.ReLU(),
        nn.Linear(200,300 ),
        # nn.BatchNorm1d(300),
        nn.ReLU(),
        nn.Linear(300,75),
        nn.ReLU(),
        nn.Linear(75,1)
    )

     # Defining the second Critic neural network
    self.features3 = nn.Sequential(
        nn.Conv2d(1 , 16 ,kernel_size = 3 ,stride = 2),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.Conv2d(16 , 32 ,kernel_size = 3 ,stride = 2),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.Conv2d(32 , 64 ,kernel_size = 3 ,stride = 2),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(output_size=(1,1))
    )

    self.linear_2 = nn.Linear(state_dim + action_dim,20)
    # self.bn_c2 = nn.BatchNorm1d(20)
    self.linear_21 = nn.Linear(20,50)
    # self.bn_c21 = nn.BatchNorm1d(50)
    self.linear_22 = nn.Linear(50,100)
    # self.bn_c22 = nn.BatchNorm1d(100)

    self.features4 = nn.Sequential(
        nn.Linear(64 + 100 ,200),
        # nn.BatchNorm1d(200),
        nn.ReLU(),
        nn.Linear(200,300 ),
        # nn.BatchNorm1d(300),
        nn.ReLU(),
        nn.Linear(300,75),
        nn.ReLU(),
        nn.Linear(75,1)
    )

  def forward(self, state_0,state_1, actor_action):

    # Forward-Propagation on the first Critic Neural Network
  
    y =  torch.cat((state_1,actor_action),dim=1)

    x1 = self.features1(state_0)
    x1 = x1.view(x1.size(0),-1)

    y1 = F.relu(self.linear_1(y))
    y1 = F.relu(self.linear_11(y1))
    y1 = F.relu(self.linear_12(y1))
    # y1 = F.relu(self.bn_c1(self.linear_1(y)))
    # y1 = F.relu(self.bn_c11(self.linear_11(y1)))
    # y1 = F.relu(self.bn_c12(self.linear_12(y1)))
    
    y1 = y1.view(y1.size(0),-1)
    merged_1 = torch.cat((x1,y1),dim=1)
    merged_1 = self.features2(merged_1)

    # Forward-Propagation on the second Critic Neural Network
    x2 = self.features3(state_0)
    x2 = x2.view(x2.size(0),-1)

    y2 = F.relu(self.linear_2(y))
    y2 = F.relu(self.linear_21(y2))
    y2 = F.relu(self.linear_22(y2))

    # y2 = F.relu(self.bn_c2(self.linear_2(y)))
    # y2 = F.relu(self.bn_c22(self.linear_22(y2)))
    y2 = y2.view(y2.size(0),-1)

    merged_2 = torch.cat((x2,y2),dim=1)
    merged_2 = self.features4(merged_2)

    return merged_1 , merged_2

  def Q1(self, state_0,state_1, actor_action):
   
    y =  torch.cat((state_1,actor_action),dim=1)

    x1 = self.features1(state_0)
    x1 = x1.view(x1.size(0),-1)

    y1 = F.relu(self.linear_1(y))
    y1 = F.relu(self.linear_11(y1))
    y1 = F.relu(self.linear_12(y1))
    # y1 = F.relu(self.bn_c1(self.linear_1(y)))
    # y1 = F.relu(self.bn_c11(self.linear_11(y1)))
    # y1 = F.relu(self.bn_c12(self.linear_12(y1)))

    y1 = y1.view(y1.size(0),-1)

    merged_1 = torch.cat((x1,y1),dim=1)
    merged_1 = self.features2(merged_1)
    return merged_1

  
## Steps 4 to 15: Training Process

# # Selecting the device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Building the whole Training Process into a class

class TD3(object):
  
  def __init__(self, state_dim,action_dim, max_action,min_action):

    self.actor = Actor( state_dim,action_dim, max_action).to(device)
    self.actor_target = Actor( state_dim,action_dim, max_action).to(device)
    self.actor_target.load_state_dict(self.actor.state_dict())
    self.actor_optimizer = torch.optim.Adam(self.actor.parameters())

    self.critic = Critic(state_dim,action_dim).to(device)
    self.critic_target = Critic(state_dim,action_dim).to(device)
    self.critic_target.load_state_dict(self.critic.state_dict())
    self.critic_optimizer = torch.optim.Adam(self.critic.parameters())
    self.max_action = max_action  
    self.min_action = min_action


  def select_action(self, single_state):
    state_01 = torch.Tensor(single_state[0]).unsqueeze(0).unsqueeze(0).to(device)
    state_02 = torch.Tensor(single_state[1]).unsqueeze(0).to(device)
    return self.actor(state_01,state_02).cpu().data.numpy().flatten()

  def train(self, replay_buffer, iterations, batch_size=100, discount=0.99, tau=0.005, policy_noise=0.2, noise_clip=0.5,
    policy_freq=2,episode_num = 1,plot_freq = 50):

    global cr_loss
    global ac_loss 
    global a_iterns
    global c_iterns
    running_critic_loss = 0.0 
    running_actor_loss  = 0.0 
    plot_loss = True

    for it in range(iterations):
      
      # Step 4: We sample a batch of transitions (s, s’, a, r) from the memory
      batch_states_0,batch_states_1, batch_next_states_0,batch_next_states_1, batch_actions, batch_rewards, batch_dones = replay_buffer.sample(batch_size)
      state_0 = torch.Tensor(batch_states_0).unsqueeze(1).to(device)
      state_1 = torch.Tensor(batch_states_1).to(device)
      next_state_0 = torch.Tensor(batch_next_states_0).unsqueeze(1).to(device)
      next_state_1 = torch.Tensor(batch_next_states_1).to(device)
      action = torch.Tensor(batch_actions).to(device)
      reward = torch.Tensor(batch_rewards).to(device)
      done = torch.Tensor(batch_dones).to(device)
      
      # Step 5: From the next state s’, the Actor target plays the next action a’
      next_action = self.actor_target(next_state_0,next_state_1)
      
      # Step 6: We add Gaussian noise to this next action a’ and we clamp it in a range of values supported by the environment
      noise = torch.Tensor(batch_actions).data.normal_(0, policy_noise).to(device)
      noise = noise.clamp(-noise_clip, noise_clip)
      next_action_noise  = (next_action + noise)
      for i in range(0,len(self.max_action)):
        next_action[:,i] = next_action_noise[:,i].clamp(self.min_action[i],self.max_action[i])
      
      # Step 7: The two Critic targets take each the couple (s’, a’) as input and return two Q-values Qt1(s’,a’) and Qt2(s’,a’) as outputs
      target_Q1, target_Q2 = self.critic_target(next_state_0,next_state_1, next_action)
      print ("target_q1_",it)
      
      # Step 8: We keep the minimum of these two Q-values: min(Qt1, Qt2)
      target_Q = torch.min(target_Q1, target_Q2)
      
      # Step 9: We get the final target of the two Critic models, which is: Qt = r + γ * min(Qt1, Qt2), where γ is the discount factor
      target_Q = reward + ((1 - done) * discount * target_Q).detach()
      
      # Step 10: The two Critic models take each the couple (s, a) as input and return two Q-values Q1(s,a) and Q2(s,a) as outputs
      current_Q1, current_Q2 = self.critic(state_0,state_1, action)
      
      # Step 11: We compute the loss coming from the two Critic models: Critic Loss = MSE_Loss(Q1(s,a), Qt) + MSE_Loss(Q2(s,a), Qt)
      critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
      # Step 12: We backpropagate this Critic loss and update the parameters of the two Critic models with a SGD optimizer
      self.critic_optimizer.zero_grad()
      critic_loss.backward()
      torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 10.0)
      self.critic_optimizer.step()

      running_critic_loss += critic_loss.item()

      # Step 13: Once every two iterations, we update our Actor model by performing gradient ascent on the output of the first Critic model
      if it % policy_freq == 0:
        actor_loss = -self.critic.Q1(state_0,state_1, self.actor(state_0,state_1)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        # print("Grad : ",  actor_loss.backward())
        self.actor_optimizer.step()

        running_actor_loss += actor_loss.item()

        # Step 14: Still once every two iterations, we update the weights of the Actor target by polyak averaging
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
          target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        
        # Step 15: Still once every two iterations, we update the weights of the Critic target by polyak averaging
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
          target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


    # Plotting Actor and Critic Losses 
      if it % plot_freq == 0 :
        cr_loss_value = running_critic_loss / plot_freq 
        ac_loss_value = running_actor_loss / (plot_freq//2)
        cr_loss.append(cr_loss_value)
        ac_loss.append(ac_loss_value)
        c_iterns = plot_freq*np.arange(0,len(cr_loss))
        a_iterns = (plot_freq//2)*np.arange(0,len(ac_loss))
        running_actor_loss = 0.0
        running_critic_loss = 0.0
        plot_loss = True

      if iterations <= plot_freq - 1 :
        cr_loss_value = running_critic_loss / iterations 
        ac_loss_value = running_actor_loss / (iterations//2)
        cr_loss.append(cr_loss_value)
        ac_loss.append(ac_loss_value)
        running_actor_loss = 0.0
        running_critic_loss = 0.0
        plot_loss = False

    # To check for exploding gradients  
    for name, param in self.critic.named_parameters():
        print("Name : ", name ," , " ,"Gradient_norm : ",param.grad.data.norm().item() ,"Gradient_max : ",param.grad.data.max().item() )
        print("Size : ",param.grad.size())

    print(cr_loss_value)
    print(ac_loss_value)

    if plot_loss :
      fig = plt.figure()
      ax1 = fig.add_subplot(2, 1, 1)
      ax2 = fig.add_subplot(2, 1, 2)
      ax1.plot(c_iterns[-100:-1],cr_loss[-100:-1],'r--')
      ax1.set_ylabel("Critic Loss")
      ax2.plot(a_iterns,ac_loss,'g--')
      ax2.set_ylabel("Actor Loss")
      ax2.set_xlabel("Iterations")
      Title = 'Episode Number :'+str(episode_num)
      fig.suptitle(Title,fontsize=16)
      plt.show()

    # Making a save method to save a trained model
  def save(self, filename, directory):
    torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename))
    torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, filename))
  
  # Making a load method to load a pre-trained model
  def load(self, filename, directory):
    self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename)))
    self.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, filename)))

action_space = gym.spaces.box.Box(low = np.array([-5,-.5]), high = np.array([5,1]), dtype=np.float32) # Rotation,Acceleration , Braking (Deceleration) 
state_dim = 6 # pos_x,pos_y,velocity_x,velocity_y,orientation,-orientation
action_dim = action_space.shape[0]
max_action = action_space.high
min_action = action_space.low
policy = TD3(state_dim,action_dim,max_action,min_action)

