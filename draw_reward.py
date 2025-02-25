from matplotlib import pyplot as plt
import numpy as np
from matplotlib import rcParams
rcParams['font.family'] = 'SimHei'
rcParams['axes.unicode_minus']=False

reward_list = np.loadtxt('./record/m2_Train_reward.txt')
# print(reward_list)
time = 20
new_list = []
for t in range(time-2,len(reward_list)):
    if(t%time==0):
        new_list.append(np.mean(reward_list[t-time:t]))
# print(new_list)
# plt.figure()
# plt.plot(range(len(reward_list)),reward_list)
# plt.title('Train_reward')
# plt.xlabel('epochs')
# plt.ylabel('reward')
# plt.show()
plt.figure()
plt.plot(range(0,len(reward_list)),reward_list)
plt.plot(range(0,len(new_list)*time,time),new_list,color='blue')
plt.title('训练过程奖励函数曲线')
plt.xlabel('训练轮次')
plt.ylabel('奖励值')
plt.savefig('./record/m2_Train_reward_new.png')