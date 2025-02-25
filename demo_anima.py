import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
fig = plt.figure()
ax = fig.subplots()

t=np.linspace(0,10,100)
y=np.sin(t)
ax.set_aspect(3)
ax.plot(t,y,'--',c='gray')
line=ax.plot(t,y,c='C2')
print(line)

def update(i):  #帧更新函数
    global t    #直接引用全局变量，也可以通过函数的frames或fargs参数传递。
    t+=0.1
    y=np.sin(t)
    line[0].set_ydata(y)
    # print(line)
    return line

ani=FuncAnimation(fig,update,interval=100) #绘制动画
plt.show() #显示动画
