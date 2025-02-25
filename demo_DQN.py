# DQN demo
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# using gym 0.23.0

# torch information

import torch
print('*'*100)
print(torch.__version__)
print('GPU: '+ str(torch.cuda.is_available()) )
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device: '+ str(device))
print('*'*100)

# gym env import

import gym
env = gym.make('CartPole-v1')

# hyper parameter
eps_final = 0.01
eps_decay = 1000
replay_batchsize = 32
Buffer_length = 1000
gamma = 0.99
learning_rate = 1e-4
epochs = 1000
val_episode = 100

# DQN pytorch model
import torch.nn as nn

input_shape = env.observation_space.shape[0]
output_shape = env.action_space.n

print("Input:{}  Output:{}".format(input_shape,output_shape))
model = nn.Sequential(
        nn.Linear(input_shape, 256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, output_shape)
        )
model = model.to(device)

# action selection
import random
def act(model, state, epsilon):
    if random.random() > epsilon: 
        state = torch.FloatTensor(np.array(state)).unsqueeze(0).to(device)
        q_value = model.forward(state)
        action = q_value.max(1)[1].item()
    else: 
        action = random.randrange(env.action_space.n)
    return action

# descending Epsilon DEF
import math
def calc_epsilon(t, epsilon_start=1.0,epsilon_final=eps_final, epsilon_decay=eps_decay):
    epsilon = epsilon_final + (epsilon_start - epsilon_final) \
            * math.exp(-1. * t / epsilon_decay)
    return epsilon

# Buffer
import numpy as np
from collections import deque

batch_size = replay_batchsize

class ReplayBuffer(object):
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        state, action, reward, next_state, done = zip( \
                *random.sample(self.buffer, batch_size))
        concat_state = np.concatenate(state)
        concat_next_state = np.concatenate(next_state)
        return concat_state, action, reward, concat_next_state, done
    
    def __len__(self):
        return len(self.buffer)

replay_buffer = ReplayBuffer(Buffer_length)

# Optimizer
import torch.optim
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)

episode_rewards = [] # 各局得分,用来判断训练是否完成
loss_tist = []
t = 0 # 训练步数,用于计算epsilon
loss = 0

# Training
for epoch in range(epochs):

    # 开始新的一局
    state = env.reset()
    episode_reward = 0

    while True:
        # env.render()
        epsilon = calc_epsilon(t)
        action = act(model, state, epsilon)
        next_state, reward, done, info = env.step(action)
        # print(state)
        replay_buffer.push(state, action, reward, next_state, done)

        state = next_state
        episode_reward += reward

        if len(replay_buffer) > batch_size:

            # 计算td error
            sample_state, sample_action, sample_reward, sample_next_state, \
                    sample_done = replay_buffer.sample(batch_size)

            sample_state = torch.tensor(sample_state, dtype=torch.float32).to(device)
            sample_action = torch.tensor(sample_action, dtype=torch.int64).to(device)
            sample_reward = torch.tensor(sample_reward, dtype=torch.float32).to(device)
            sample_next_state = torch.tensor(sample_next_state,dtype=torch.float32).to(device)
            sample_done = torch.tensor(sample_done, dtype=torch.float32).to(device)
            
            next_qs = model(sample_next_state)
            next_q, _ = next_qs.max(1)
            expected_q = sample_reward + gamma * next_q * (1 - sample_done)
            
            qs = model(sample_state)
            q = qs.gather(1, sample_action.unsqueeze(1)).squeeze(1)
            
            td_error = expected_q - q
            
            # 计算 MSE 损失
            loss = td_error.pow(2).mean() 
            
            # 根据损失改进网络
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            t += 1
            
        if done: # 本局结束
            i_episode = len(episode_rewards)
            print('Epi:{} Reward = {:.4f} loss = {:.4f} eps = {:.4f}'.format(i_episode,episode_reward,loss,epsilon))
            # print('End_state:',state)
            # print('Leader&follower:',info['leader_pos'],info['follower_pos'],info['time'])
            # print(' ')
            episode_rewards.append(float(episode_reward))
            loss_tist.append(float(loss))
            break
            
    # if len(episode_rewards) > 100 and np.mean(episode_rewards[-30:]) > 490:
    #     break # 训练结束

# validation
n_episode = val_episode
val_reward_list = []
for i_episode in range(n_episode):

    observation = env.reset()
    episode_reward = 0
    while True:
        # if i_episode%5==0:
        #     env.render()
        action = act(model, observation, 0)
        observation, reward, done, info = env.step(action)
        episode_reward += reward
        state = observation
        if done:
            break
    print('Val_Epi:{} Reward = {:.4f}'.format(i_episode, episode_reward))
    # print('End_state:',state)
    # print('Leader&follower:',info['leader_pos'],info['follower_pos'],info['time'])
    print(' ')
    val_reward_list.append(episode_reward)

from matplotlib import pyplot as plt
# print(loss_tist)
# plt.figure()
# plt.plot(range(len(loss_tist)),loss_tist)
# plt.title('loss')
# plt.ylim(0,0.1)
# plt.show()

plt.figure()
plt.plot(range(len(episode_rewards)),episode_rewards)
plt.title('Train_reward')
plt.show()

plt.figure()
plt.plot(range(len(val_reward_list)),val_reward_list)
plt.title('Val_reward')
plt.show()
