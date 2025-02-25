import gym
env = gym.make('CarFollow1')
env.reset()
xlist = []
ylist = []
for i in range(400):
    # env.render()
    next_state, reward, done, info = env.step(env.action_space.sample()) # take a random action
    x = info['leader_state'][0]
    y =info['leader_state'][1]
    xlist.append(x)
    ylist.append(y)
    # if done:
    #     
    #     break
env.close()
print(i)
from matplotlib import pyplot as plt

plt.figure()
ax = plt.gca()
ax.set_aspect(1)
plt.plot(xlist,ylist)
# plt.xlim(-6,6)
# plt.ylim(-6,6)
plt.show()
