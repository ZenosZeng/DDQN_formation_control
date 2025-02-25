# Mission 1 Train
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from matplotlib import pyplot as plt

# using gym 0.23.0

#-------------------------------------------------------------------------------------
# torch information
import torch
print('*'*100)
print(torch.__version__)
print('GPU: '+ str(torch.cuda.is_available()) )
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device: '+ str(device))
print('*'*100)

#-------------------------------------------------------------------------------------
# gym env import
import gym
env = gym.make('CarFollow-v0')

#-------------------------------------------------------------------------------------
# hyper parameter
eps_final = 0.01
eps_start = 1.0
eps_decay_end = 3000
replay_batchsize = 512
Buffer_length = int(2**20)
gamma = 0.95
learning_rate = 1e-4/2
epochs = 8000
val_episode = 10
tau = 0.005

#-------------------------------------------------------------------------------------
# DDQN pytorch model
import torch.nn as nn
import numpy as np

input_shape = env.observation_space.shape[0]
output_shape = env.action_space.n

print("Input:{}  Output:{}".format(input_shape,output_shape))
print('*'*50)

New_model = True

if(New_model==True):
    model = nn.Sequential(
            nn.Linear( input_shape, 256 ),
            nn.ReLU(),
            nn.Linear( 256 ,  256*2 ),
            nn.ReLU(),
            nn.Linear( 256*2 ,  256 ),
            nn.ReLU(),
            nn.Linear( 256 , output_shape )
            )
    t_model = model
    t_model.load_state_dict(model.state_dict())
    episode_rewards = [] #  reward list for each episode
else:
    model = torch.load('./record/m1_DDQN_net_finish')
    t_model = torch.load('./record/m1_DDQN_tnet_finish')
    episode_rewards = list( np.loadtxt('./record/m1_Train_reward.txt') )
    eps_start=eps_final

model = model.to(device)
t_model = t_model.to(device)

#-------------------------------------------------------------------------------------
# checkpoint
def checkpoint(model,model2,reward_list):
    # save net
    torch.save(model,'./record/checkpoint/m1_DDQN_net_'+str(len(reward_list)))
    torch.save(model2,'./record/checkpoint/m1_DDQN_tnet_'+str(len(reward_list)))
    # plot reward
    plt.figure()
    plt.plot(range(len(reward_list)),reward_list)
    plt.title('Train_rewards')
    plt.xlabel('epochs')
    plt.ylabel('reward')
    plt.savefig('./record/checkpoint/m1_Train_reward_'+str(len(reward_list))+'_epoch'+'.png')
    plt.close()
    # save reward
    np.savetxt('./record/checkpoint/m1_Train_reward_'+str(len(reward_list))+'_epoch'+'.txt',np.array(reward_list))

#-------------------------------------------------------------------------------------
# action selection
import random
random.seed(42)
def act(model, state, epsilon, ranS=True):
    if(ranS==False):
        state = torch.FloatTensor(np.array(state)).unsqueeze(0).to(device)
        q_value = model.forward(state)
        action = q_value.max(1)[1].item()
        return action
    else:
        if random.random() > epsilon: 
            state = torch.FloatTensor(np.array(state)).unsqueeze(0).to(device)
            q_value = model.forward(state)
            action = q_value.max(1)[1].item()
        else: 
            action = random.randrange(env.action_space.n)
            
    return action

#-------------------------------------------------------------------------------------
# descending Epsilon definition
import math
def calc_epsilon(epi, epsilon_start=eps_start,epsilon_final=eps_final, ending=eps_decay_end):
    if epi<ending:
        return (epsilon_start - epsilon_final)*(ending-epi)/ending
    else:
        return epsilon_final
    
#-------------------------------------------------------------------------------------
# Buffer

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

#-------------------------------------------------------------------------------------
# Optimizer
import torch.optim
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)

#-------------------------------------------------------------------------------------
# Training

loss_tist = []
t = 0 # for epsilon 
loss = 0
for epoch in range(epochs):

    # new episode
    state = env.reset()
    episode_reward = 0

    while True:
        # select action
        i_episode = len(episode_rewards)
        epsilon = calc_epsilon(i_episode)
        action = act(model, state, epsilon)

        # step
        next_state, reward, done, info = env.step(action)

        # exit if done
        if done: 
            i_episode = len(episode_rewards)
            print('Epi:{} Reward = {:.4f} loss = {:.4f} eps = {:.4f}'.format(i_episode,episode_reward,loss,epsilon))
            print('End_state:',state)
            print('Leader_state:',info['leader_state'])
            print('Follower_state:',info['follower_state'])
            print('End_time:',info['time'])
            print('*'*50)
            episode_rewards.append(float(episode_reward))
            loss_tist.append(float(loss))
            if( (i_episode+1)%100==0 and i_episode>0):
                checkpoint(model,t_model,episode_rewards)
            break

        # save experience
        replay_buffer.push(state, action, reward, next_state, done)
        state = next_state
        # add reward
        episode_reward += reward

        # update NETWORK
        if len(replay_buffer) > batch_size:

            # td error
            sample_state, sample_action, sample_reward, sample_next_state, \
                    sample_done = replay_buffer.sample(batch_size)

            sample_state = torch.tensor(sample_state, dtype=torch.float32).to(device)
            sample_action = torch.tensor(sample_action, dtype=torch.int64).to(device)
            sample_reward = torch.tensor(sample_reward, dtype=torch.float32).to(device)
            sample_next_state = torch.tensor(sample_next_state,dtype=torch.float32).to(device)
            batch_idx = torch.arange(batch_size, dtype=torch.long).to(device)
            # sample_done = torch.tensor(sample_done, dtype=torch.float32).to(device)
            
            with torch.no_grad():
                next_qs = t_model.forward(sample_next_state)
                # next_q, _ = next_qs.max(1)
                max_actions = torch.argmax(model.forward(sample_next_state),dim=-1)
                next_q = next_qs[batch_idx,max_actions]
                expected_q = sample_reward + gamma * next_q
            qs = model(sample_state)
            q = qs.gather(1, sample_action.unsqueeze(1)).squeeze(1)
            td_error = expected_q - q
            loss = td_error.pow(2).mean() 

            # optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # soft update target net
            for t_model_params, model_params in zip(t_model.parameters(), model.parameters()):
                t_model_params.data.copy_(tau * model_params + (1 - tau) * t_model_params)

#-------------------------------------------------------------------------------------
# validation
n_episode = val_episode
val_reward_list = []
for i_episode in range(n_episode):

    state = env.reset()
    episode_reward = 0
    while True:
        # if i_episode%5==0:
        #     env.render()
        if(i_episode<n_episode/2):
            action = act(model, state, 0,  ranS=False)
        else:
            action = act(t_model, state, 0,  ranS=False)
        next_state, reward, done, info = env.step(action)
        episode_reward += reward
        state = next_state
        if done:
            break
    print('Val_Epi:{} Reward = {:.4f}'.format(i_episode, episode_reward))
    print('End_state:',state)
    print('Leader_state:',info['leader_state'])
    print('Follower_state:',info['follower_state'])
    print('End_time:',info['time'])
    print('*'*50)
    val_reward_list.append(episode_reward)

#-------------------------------------------------------------------------------------
# save NETWORK
torch.save(model,'./record/m1_DDQN_net_finish')
torch.save(t_model,'./record/m1_DDQN_tnet_finish')
np.savetxt('./record/checkpoint/m1_Train_reward.txt',np.array(episode_rewards))

#-------------------------------------------------------------------------------------
# draw reward figure and save
plt.close()
plt.figure()
plt.plot(range(len(episode_rewards)),episode_rewards,label='DDQN')
plt.title('Train_reward')
plt.xlabel('epochs')
plt.ylabel('reward')
plt.savefig('./record/m1_Train_reward.png')

plt.figure()
plt.plot(range(len(val_reward_list)),val_reward_list,label='DDQN')
plt.title('Val_reward')
plt.xlabel('epochs')
plt.ylabel('reward')
plt.savefig('./record/m1_Val_reward.png')

# display
plt.show()

