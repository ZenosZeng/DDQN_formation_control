# Mission 2 eval
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

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
env = gym.make('CarFollow1')

#-------------------------------------------------------------------------------------
# D3QN pytorch model LOADING
import torch.nn as nn

model_file_name= 'm2_DDQN_net'
model = torch.load(model_file_name)
model = model.to(device)

#-------------------------------------------------------------------------------------
# action selection
import numpy as np
import random
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
# evaluation

rwdlist = []

for epoch in range(100):
    leader_x_list = []
    leader_y_list = []
    leader_theta_list = []
    follower_x_list = []
    follower_y_list = []
    follower_theta_list = []
    coor_list = []

    for i_episode in range(1):

        state = env.reset()
        episode_reward = 0
        while True:
            action = act(model, state, 0,ranS=False)
            next_state, reward, done, info = env.step(action)

            leader_x_list.append(info['leader_state'][0])
            leader_y_list.append(info['leader_state'][1])
            leader_theta_list.append(info['leader_state'][2])
            follower_x_list.append(info['follower_state'][0])
            follower_y_list.append(info['follower_state'][1])
            follower_theta_list.append(info['follower_state'][2])
            coor_list.append([info['leader_state'][0],info['leader_state'][1],info['leader_state'][2],
                            info['follower_state'][0],info['follower_state'][1],info['follower_state'][2]])

            episode_reward += reward
            state = next_state
            if done:
                break
        
        print('Reward = {:.4f}'.format(episode_reward))
        rwdlist.append(episode_reward)
        # print('End_state:',state)
        # print('Leader_state:',info['leader_state'])
        # print('Follower_state:',info['follower_state'])
        print('End_time:',info['time'])
        print(epoch)
        print('*'*50)
print(rwdlist)
from matplotlib import pyplot as plt
from math import cos,sin

# reward_list = np.loadtxt('m2_Train_reward.txt')
# print(reward_list)
time = 3
time2 = 10
new_list = []
new_list2 = []
for t in range(0,len(rwdlist)):
    new_list.append(np.mean(rwdlist[max(0,t-time):t]))
    new_list2.append(np.mean(rwdlist[max(0,t-time2):t]))

from matplotlib import rcParams
rcParams['font.family'] = 'SimHei'
rcParams['axes.unicode_minus']=False
plt.figure()
plt.plot(range(0,len(rwdlist)),rwdlist,color='grey',linewidth=0.5,label='原始值')
plt.plot(range(0,len(new_list)),new_list,color='green',linewidth=0.8,label='k=5')
plt.plot(range(0,len(new_list2)),new_list2,color='red',linewidth=1,label='k=30')
plt.title('验证过程奖励函数曲线')
plt.xlabel('验证轮次')
plt.ylabel('奖励值')
plt.ylim(0,100000)
plt.legend()
plt.savefig('m2_eval_reward_new.png')

# #-------------------------------------------------------------------------------------
# # ploting

# def draw_car(x,y,theta):
#     L=0.1
#     LL=0.2
#     x1 = x + (LL-L)*cos(theta) - L*sin(theta)
#     y1 = y + L*cos(theta) + (LL-L)*sin(theta)

#     x2 = x - LL*cos(theta) - L*sin(theta)
#     y2 = y + L*cos(theta) - LL*sin(theta)

#     x3 = x - LL*cos(theta) + L*sin(theta)
#     y3 = y - L*cos(theta) - LL*sin(theta)

#     x4 = x + (LL-L)*cos(theta) + L*sin(theta)
#     y4 = y - L*cos(theta) + (LL-L)*sin(theta)

#     x5 = x + (LL)*cos(theta) 
#     y5 = y + (LL)*sin(theta)

#     return [x1,x2,x3,x4,x5,x1],[y1,y2,y3,y4,y5,y1]


# # trajectory 4
# plt.figure(dpi=100)

# plt.plot(leader_x_list,leader_y_list,color='red',label='Leader',linewidth=0.5)
# plt.plot(follower_x_list,follower_y_list,color='green',label='Follower',linewidth=0.5)
# # plt.scatter(leader_x_list[-1],leader_y_list[-1],color='red',marker='x',linewidth=0.8)                # ,marker='x',linewidth='1'
# # plt.scatter(follower_x_list[-1],follower_y_list[-1],color='green',marker='x',linewidth=0.8)
# plt.scatter(leader_x_list[0],leader_y_list[0],color='red',marker='x',linewidth=0.5) 
# plt.scatter(follower_x_list[0],follower_y_list[0],color='green',marker='x',linewidth=0.5)
# x,y = draw_car(leader_x_list[-1],leader_y_list[-1],leader_theta_list[-1])
# plt.plot(x,y,color='red',linewidth=0.5)
# x,y = draw_car(follower_x_list[-1],follower_y_list[-1],follower_theta_list[-1])
# plt.plot(x,y,color='green',linewidth=0.5)


# ax = plt.gca()
# ax.set_aspect(1)
# # plt.xlim(xmin,xmax)
# # plt.ylim(ymin,ymax)
# plt.title('Car_Trajectory')
# plt.xlabel('x/m')
# plt.ylabel('y/m')
# plt.legend()
# plt.savefig('m2_eval_trajectory.png')

# # x error plot
# plt.figure(dpi=300)
# plt.subplot(211)
# # axes = plt.gca()
# # axes.spines['right'].set_color('none')
# # axes.spines['top'].set_color('none')
# # axes.spines['bottom'].set_position(('data', 0))
# # axes.spines['left'].set_position(('data', 0))
# ex_list = [ leader_x_list[i] - follower_x_list[i] for i in range(len(leader_x_list)) ] 
# plt.ylim(1.5*min(-0.1,min(ex_list)),1.5*max(0.1,max(ex_list)))
# plt.xlabel('t/s')
# plt.ylabel('x/m')
# plt.plot(np.arange(0,len(ex_list)/10,0.1),ex_list)
# plt.title('x_error')
# # plt.savefig('./record/m2_eval_x_error.png')

# # y error plot
# plt.subplot(212)
# axes = plt.gca()
# # axes.spines['right'].set_color('none')
# # axes.spines['top'].set_color('none')
# # axes.spines['bottom'].set_position(('data', 0))
# # axes.spines['left'].set_position(('data', 0))
# ey_list = [ leader_y_list[i] - follower_y_list[i] for i in range(len(leader_y_list)) ] 
# plt.ylim(1.5*min(-0.1,min(ey_list)),1.5*max(0.1,max(ey_list)))
# # plt.xlim(0,len(ex_list))
# plt.xlabel('t/s')
# plt.ylabel('y/m')
# plt.plot(np.arange(0,len(ey_list)/10,0.1),ey_list)
# plt.title('y_error')
# plt.savefig('m2_eval_xy_error.png')

#-------------------------------------------------------------------------------------
# Trajectory Animation
# from matplotlib.animation import FuncAnimation
# from matplotlib import animation

# fig = plt.figure(dpi=100)
# ax = fig.subplots()

# t=0
# ax.set_aspect(1)
# ax.set_title('Car_Trajectory')
# ax.set_xlabel('x/m')
# ax.set_ylabel('y/m')
# ax.legend()

# color_0='red'
# color_1='green'
# line_width = 0.8

# line0=ax.plot(leader_x_list[:t],leader_y_list[:t],color=color_0,label='Leader')
# line1=ax.plot(follower_x_list[:t],follower_y_list[:t],color=color_1,label='Follower')
# line2=ax.plot(leader_x_list[:t],leader_y_list[:t],color=color_0)
# line3=ax.plot(follower_x_list[:t],follower_y_list[:t],color=color_1)

# def update(time):  
#     global t    
#     # ax.set_xlim(-1,7)
#     # ax.set_ylim(-4,4)
#     ax.legend()
#     ax.set_title('Car_Trajectory {:.1f}s'.format(t/10))
#     t+=1
#     line0[0].set_data(leader_x_list[:t],leader_y_list[:t])
#     line1[0].set_data(follower_x_list[:t],follower_y_list[:t])
#     x,y = draw_car(leader_x_list[t],leader_y_list[t],leader_theta_list[t])
#     line2[0].set_data(x,y)
#     x,y = draw_car(follower_x_list[t],follower_y_list[t],follower_theta_list[t])
#     line3[0].set_data(x,y)

#     # print(line)
#     return line0,line1,line2,line3

# ani=FuncAnimation(fig,update,interval=10,frames=300-2) #绘制动画
# ani.save('m2_eval_animation.gif') #显示动画

# print('Trajectory Animation created.')
