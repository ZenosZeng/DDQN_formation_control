from matplotlib import pyplot as plt
import numpy as np
from matplotlib import rcParams
rcParams['font.family'] = 'SimHei'
rcParams['axes.unicode_minus']=False

reward_list = np.loadtxt('m2_Train_reward.txt')
# print(reward_list)
time = 5
new_list = []
for t in range(time-2,len(reward_list)):
    if(t%time==0):
        new_list.append(np.mean(reward_list[t-time:t]))
time2 = 30
new_list2 = []
for t in range(time2-2,len(reward_list)):
    if(t%time2==0):
        new_list2.append(np.mean(reward_list[t-time2:t]))

plt.figure()
plt.plot(range(0,len(reward_list)),reward_list,color='grey',linewidth=0.1,label='原始值')
plt.plot(range(0,len(new_list)*time,time),new_list,color='green',linewidth=0.8,label='k=5')
plt.plot(range(0,len(new_list2)*time2,time2),new_list2,color='red',linewidth=1,label='k=30')
plt.title('训练过程奖励函数曲线')
plt.xlabel('训练轮次')
plt.ylabel('奖励值')
plt.legend()
plt.savefig('m2_Train_reward_new.png')