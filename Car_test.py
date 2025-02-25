import CarNew
import numpy as np
from matplotlib import pyplot as plt
from math import cos,sin

plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False

from math import pi
car1 = CarNew.Car()
car2 = CarNew.Car()
# car1.reset(0,-1,0,v=0.2)
car2.reset(0,0,0,v=0.2)
xlist_1 = []
ylist_1 = []
xlist_2 = []
ylist_2 = []
thetalist = []
tlist = np.arange(0,1200,1)

acc2 = 0.01
leaderU = [ (0,0) for t in range(len(tlist)) ]
for t in tlist:
    if 110<t<120:
        leaderU[t]=(-0.4*0.001,0.4*0.001)
    elif 460<t<470:
        leaderU[t]=(0.4*0.001,-0.4*0.001)
    if 600<t<610:
        leaderU[t]=(0.4*0.001,-0.4*0.001)
    elif 950<t<960:
        leaderU[t]=(-0.4*0.001,0.4*0.001)
    else:
        pass

    car2.run(leaderU[t][0],leaderU[t][1])
    # print(car1.x,car1.y,car1.theta)
    xlist_1.append(car1.x)
    ylist_1.append(car1.y)
    xlist_2.append(car2.x)
    ylist_2.append(car2.y)
    thetalist.append(car2.theta)

plt.figure()
plt.plot(xlist_2,ylist_2,color='black')
# plt.plot(xlist_2[:40],ylist_2[:40],color='blue',label='匀速段')
# plt.plot(xlist_2[40:80],ylist_2[40:80],color='orange',label='右转段')
# plt.plot(xlist_2[80:120],ylist_2[80:120],color='red',label='加减速段')
# plt.plot(xlist_2[120:160],ylist_2[120:160],color='green',label='左转段')
# plt.plot(xlist_2[160:],ylist_2[160:],color='blue')

ax = plt.gca()
ax.set_aspect(1)
# for i in range(len(xlist_2)):
    # plt.plot([xlist_2[i],xlist_2[i]+0.1*cos(thetalist[i])],[ylist_2[i],ylist_2[i]+0.1*sin(thetalist[i])],color='r',linewidth=0.1)
plt.xlim(-2,12)
plt.ylim(-7,7)
plt.title('领导者轨迹')
plt.xlabel('x/m')
plt.ylabel('y/m')
# plt.legend()
plt.show()

# plt.figure()
# plt.plot(tlist,thetalist)
# plt.show()

