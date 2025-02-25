# Mission 1 eval 3
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# using gym 0.23.0

#-------------------------------------------------------------------------------------
# torch information
import torch
# print('*'*100)
# print(torch.__version__)
# print('GPU: '+ str(torch.cuda.is_available()) )
# device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print('device: '+ str(device))
# print('*'*100)

#-------------------------------------------------------------------------------------
# gym env import
# import gym
# env = gym.make('CarFollow-v0')

#-------------------------------------------------------------------------------------
# DDQN pytorch model LOADING

model_file_name= 'm1_DDQN_net_finish'
model = torch.load('./record/'+model_file_name)
model = model.to('cpu')

print('model Loaded.')

#-------------------------------------------------------------------------------------
# action selection
import numpy as np

'''
Action need to be copied from gym.env with training
'''

def act(model, state):
    state = torch.FloatTensor(np.array(state)).unsqueeze(0)
    q_value = model.forward(state)
    action = q_value.max(1)[1].item()
    
    acc = 0.01
    dec = -0.01
    turnning = 0.001

    if action == 0: # maintain
        ur = 0
        ul = 0
    elif action == 1: # acclerate
        ur = acc
        ul = acc
    elif action == 2: # brake
        ur = dec
        ul = dec
    elif action == 3: # turn left
        ur = turnning
        ul = -turnning
    elif action == 4: # turn right
        ur = -turnning
        ul = turnning
    elif action == 5: # turn left + acc
        ur = turnning + acc
        ul = -turnning + acc
    elif action == 6: # turn left + brake
        ur = turnning + dec
        ul = -turnning + dec
    elif action == 7: # turn right + acc
        ur = -turnning + acc
        ul = turnning + acc
    elif action == 8: # turn right + brake
        ur = -turnning + dec
        ul = turnning + dec
    else:
        raise Exception("Invalid action")
    
    return ur,ul

#-------------------------------------------------------------------------------------
# evaluation Env and Data Collector

'''
COM: 0-1&2 1-3 2-4
coordinate:5 x y theta v w
coor_list[i][j][k]
    i: coordinate from 0-4: x y theta v w
    j: Car No.
    k: time
'''

# data list init
from CarNew import Car
coordinate_total=5
car_total = 5
time_total = 300
Com_line = 4

car=[ Car() for i in range(car_total) ]
coor_list = [ [ [  ] for i in range(coordinate_total) ] for i in range(car_total) ]

print('Env ready.')

#-------------------------------------------------------------------------------------
# initization

# reset pos and velo
from math import pi
car[0].reset(0,0,0,v=0.2)
car[1].reset(0,1.5,0)
car[2].reset(0,-1.5,0)
car[3].reset(0,3,0)
car[4].reset(0,-3,0)

for i in range(car_total):
    coor_list[i][0].append(car[i].x)
    coor_list[i][1].append(car[i].y)
    coor_list[i][2].append(car[i].theta)
    coor_list[i][3].append(car[i].v)
    coor_list[i][4].append(car[i].w)

# error intergral init
ei_list = [ [0,0] for i in range(Com_line)] # 01 02 13 24

print('All car initialized.')

#-------------------------------------------------------------------------------------
# run the car
from math import cos,sin
'''
Training State:
    ex =  self.Leader.x - self.Follower.x
    ey =  self.Leader.y - self.Follower.y
    self.iex += ex/10
    self.iey += ey/10
    current_state = np.array([ex,ey,self.iex,self.iey,\
                                    self.Follower.v*cos(theta_f),self.Follower.v*sin(theta_f),self.Follower.w,\
                                    self.Leader.v*cos(theta_l),self.Leader.v*sin(theta_l),self.Leader.w])
'''

# run the Env
for t in range(time_total):

    # update leader 0
    if t<80:
        car[0].run(0,0)
    elif 80<t<100:
        car[0].run(-0.2*0.001,0.2*0.001)
    elif 100<t<120:
        car[0].run(0.2*0.001,-0.2*0.001)
    elif 120<t<140:
        car[0].run(0.01,0.01)    
    elif 140<t<160:
        car[0].run(-0.01,-0.01)
    elif 160<t<180:
        car[0].run(0.2*0.001,-0.2*0.001)
    elif 180<t<200:
        car[0].run(-0.2*0.001,0.2*0.001)
    # elif 250<t<260:
    #     car[0].run(-0.5*0.001,0.5*0.001)
    # elif 260+270<t<270+270:
    #     car[0].run(0.5*0.001,-0.5*0.001)
    else:
        car[0].run(0,0)
    # update follower 1
    ex01 = car[0].x-car[1].x-1
    ey01 = car[0].y-car[1].y+1
    ei_list[0][0] += ex01/10
    ei_list[0][1] += ey01/10
    s01 = [ex01,ey01,ei_list[0][0],ei_list[0][1],\
        car[1].v*cos(car[1].theta),car[1].v*sin(car[1].theta),car[1].w,\
        car[0].v*cos(car[0].theta),car[0].v*sin(car[0].theta),car[0].w] 
    ur1,ul1 = act(model,s01)
    car[1].run(ur1,ul1)

    # update follower 2
    ex02 = car[0].x-car[2].x-1
    ey02 = car[0].y-car[2].y-1
    ei_list[1][0] += ex02/10
    ei_list[1][1] += ey02/10
    s02 = [ex02,ey02,ei_list[1][0],ei_list[1][1],\
        car[2].v*cos(car[2].theta),car[2].v*sin(car[2].theta),car[2].w,\
        car[0].v*cos(car[0].theta),car[0].v*sin(car[0].theta),car[0].w] 
    ur2,ul2 = act(model,s02)
    car[2].run(ur2,ul2)

    # update follower 3
    ex13 = car[1].x-car[3].x-1
    ey13 = car[1].y-car[3].y
    ei_list[2][0] += ex13/10
    ei_list[2][1] += ey13/10
    s13 = [ex13,ey13,ei_list[2][0],ei_list[2][1],
        car[3].v*cos(car[3].theta),car[3].v*sin(car[3].theta),car[3].w,\
        car[1].v*cos(car[1].theta),car[1].v*sin(car[1].theta),car[1].w] 
    ur3,ul3 = act(model,s13)
    car[3].run(ur3,ul3)

    # update follower 4
    ex24 = car[2].x-car[4].x-1
    ey24 = car[2].y-car[4].y
    ei_list[3][0] += ex24/10
    ei_list[3][1] += ey24/10
    s24 = [ex24,ey24,ei_list[3][0],ei_list[3][1],\
        car[4].v*cos(car[4].theta),car[4].v*sin(car[4].theta),car[4].w,\
        car[2].v*cos(car[2].theta),car[2].v*sin(car[2].theta),car[2].w] 
    ur4,ul4 = act(model,s24)
    car[4].run(ur4,ul4)

    # save data of all car
    for i in range(car_total):
        coor_list[i][0].append(car[i].x)
        coor_list[i][1].append(car[i].y)
        coor_list[i][2].append(car[i].theta)
        coor_list[i][3].append(car[i].v)
        coor_list[i][4].append(car[i].w)

# print(coor_list[0][0])
print('Run Completed & Data Collected.')

#-------------------------------------------------------------------------------------
# Trajectory Ploting
from matplotlib import pyplot as plt
plt.figure(dpi=800)

# car0                         ,marker='x',linewidth='1'
color_0 = 'red'
line_width = 0.8
plt.plot(coor_list[0][0],coor_list[0][1],color=color_0,label='Leader',linewidth=line_width)
plt.scatter(coor_list[0][0][0],coor_list[0][1][0],color=color_0,linewidth=line_width)
plt.scatter(coor_list[0][0][-1],coor_list[0][1][-1],color=color_0,marker='x',linewidth=line_width)                
# car1
color_1 = 'green'
plt.plot(coor_list[1][0],coor_list[1][1],color=color_1,label='Follower1',linewidth=line_width)
plt.scatter(coor_list[1][0][0],coor_list[1][1][0],color=color_1,linewidth=line_width)
plt.scatter(coor_list[1][0][-1],coor_list[1][1][-1],color=color_1,marker='x',linewidth=line_width)
# car2
color_2 = 'blue'
plt.plot(coor_list[2][0],coor_list[2][1],color=color_2,label='Follower2',linewidth=line_width)
plt.scatter(coor_list[2][0][0],coor_list[2][1][0],color=color_2,linewidth=line_width)
plt.scatter(coor_list[2][0][-1],coor_list[2][1][-1],color=color_2,marker='x',linewidth=line_width)
# car3
color_3 = 'purple'
plt.plot(coor_list[3][0],coor_list[3][1],color=color_3,label='Follower3',linewidth=line_width)
plt.scatter(coor_list[3][0][0],coor_list[3][1][0],color=color_3,linewidth=line_width)
plt.scatter(coor_list[3][0][-1],coor_list[3][1][-1],color=color_3,marker='x',linewidth=line_width)
# car4
color_4 = 'orange'
plt.plot(coor_list[4][0],coor_list[4][1],color=color_4,label='Follower4',linewidth=line_width)
plt.scatter(coor_list[4][0][0],coor_list[4][1][0],color=color_4,linewidth=line_width)
plt.scatter(coor_list[4][0][-1],coor_list[4][1][-1],color=color_4,marker='x',linewidth=line_width)

# axe
ax = plt.gca()
ax.set_aspect(1)
plt.ylim(-6,6)
plt.xlim(-3,9)
plt.title('Car_Trajectory')
plt.xlabel('x/m')
plt.ylabel('y/m')
plt.legend()
plt.savefig('./record/m1_eval_3a_trajectory.png')

print('Trajectory Ploted.')

#-------------------------------------------------------------------------------------
# Error Ploting

# x 
plt.figure(dpi=800)
plt.subplot(211)
axes = plt.gca()
axes.spines['right'].set_color('none')
axes.spines['top'].set_color('none')
axes.spines['bottom'].set_position(('data', 0))
# axes.spines['left'].set_position(('data', 0))
ex01_list = [ coor_list[0][0][i] - coor_list[1][0][i] for i in range(time_total) ] 
ex02_list = [ coor_list[0][0][i] - coor_list[2][0][i] for i in range(time_total) ]
ex03_list = [ coor_list[0][0][i] - coor_list[3][0][i] for i in range(time_total) ] 
ex04_list = [ coor_list[0][0][i] - coor_list[4][0][i] for i in range(time_total) ]  
plt.ylim(-5,5)
plt.xlabel('t/s')
plt.ylabel('x/m')
plt.plot(np.arange(0,time_total/10,0.1),ex01_list,color=color_1,label='Follower1')
plt.plot(np.arange(0,time_total/10,0.1),ex02_list,color=color_2,label='Follower2')
plt.plot(np.arange(0,time_total/10,0.1),ex03_list,color=color_3,label='Follower3')
plt.plot(np.arange(0,time_total/10,0.1),ex04_list,color=color_4,label='Follower4')
plt.title('x_error')

# y
# plt.figure(dpi=800)
plt.subplot(212)
axes = plt.gca()
axes.spines['right'].set_color('none')
axes.spines['top'].set_color('none')
axes.spines['bottom'].set_position(('data', 0))
# axes.spines['left'].set_position(('data', 0))
ey01_list = [ coor_list[0][1][i] - coor_list[1][1][i] for i in range(time_total) ]  # np.arange(0,time_total/10,0.1)
ey02_list = [ coor_list[0][1][i] - coor_list[2][1][i] for i in range(time_total) ]
ey03_list = [ coor_list[0][1][i] - coor_list[3][1][i] for i in range(time_total) ] 
ey04_list = [ coor_list[0][1][i] - coor_list[4][1][i] for i in range(time_total) ]  
plt.ylim(-5,5)
plt.xlabel('t/s')
plt.ylabel('y/m')
plt.plot(np.arange(0,time_total/10,0.1),ey01_list,color=color_1,label='Follower1')
plt.plot(np.arange(0,time_total/10,0.1),ey02_list,color=color_2,label='Follower2')
plt.plot(np.arange(0,time_total/10,0.1),ey03_list,color=color_3,label='Follower3')
plt.plot(np.arange(0,time_total/10,0.1),ey04_list,color=color_4,label='Follower4')
plt.title('y_error')
# plt.legend()
plt.savefig('./record/m1_eval_3a_xy_error.png')

# finish
print('Error Ploted.')

#-------------------------------------------------------------------------------------
# Trajectory Animation
from matplotlib.animation import FuncAnimation
from matplotlib import animation

fig = plt.figure()
ax = fig.subplots()

t=0
ax.set_aspect(1)
ax.set_title('Car_Trajectory')
ax.set_xlabel('x/m')
ax.set_ylabel('y/m')
ax.legend()
line=[ 0 for i in range(2*car_total) ]
line[0]=ax.plot(coor_list[0][0][:t],coor_list[0][1][:t],color=color_0,label='Leader')
line[1]=ax.plot(coor_list[1][0][:t],coor_list[1][1][:t],color=color_1,label='Follower1')
line[2]=ax.plot(coor_list[2][0][:t],coor_list[2][1][:t],color=color_2,label='Follower2')
line[3]=ax.plot(coor_list[3][0][:t],coor_list[3][1][:t],color=color_3,label='Follower3')
line[4]=ax.plot(coor_list[4][0][:t],coor_list[4][1][:t],color=color_4,label='Follower4')

line[0+car_total]=ax.plot(coor_list[0][0][:t],coor_list[0][1][:t],color=color_0,linewidth=line_width)
line[1+car_total]=ax.plot(coor_list[1][0][:t],coor_list[1][1][:t],color=color_1,linewidth=line_width)
line[2+car_total]=ax.plot(coor_list[2][0][:t],coor_list[2][1][:t],color=color_2,linewidth=line_width)
line[3+car_total]=ax.plot(coor_list[3][0][:t],coor_list[3][1][:t],color=color_3,linewidth=line_width)
line[4+car_total]=ax.plot(coor_list[4][0][:t],coor_list[4][1][:t],color=color_4,linewidth=line_width)

def draw_car(x,y,theta):
    L=0.1
    LL=0.2
    x1 = x + (LL-L)*cos(theta) - L*sin(theta)
    y1 = y + L*cos(theta) + (LL-L)*sin(theta)

    x2 = x - LL*cos(theta) - L*sin(theta)
    y2 = y + L*cos(theta) - LL*sin(theta)

    x3 = x - LL*cos(theta) + L*sin(theta)
    y3 = y - L*cos(theta) - LL*sin(theta)

    x4 = x + (LL-L)*cos(theta) + L*sin(theta)
    y4 = y - L*cos(theta) + (LL-L)*sin(theta)

    x5 = x + (LL)*cos(theta) 
    y5 = y + (LL)*sin(theta)

    return [x1,x2,x3,x4,x5,x1],[y1,y2,y3,y4,y5,y1]

def update(i):  #帧更新函数
    global t,car_total    #直接引用全局变量，也可以通过函数的frames或fargs参数传递。
    ax.set_xlim(-3,9)
    ax.set_ylim(-6,6)
    ax.legend()

    t+=1
    for i in range(car_total):
        x,y = draw_car(coor_list[i][0][t],coor_list[i][1][t],coor_list[i][2][t])
        line[i][0].set_data(x,y)
    for i in range(car_total,car_total*2):
        line[i][0].set_data(coor_list[i-car_total][0][:t],coor_list[i-car_total][1][:t])

    # print(line)
    return line

ani=FuncAnimation(fig,update,interval=30,frames=300-1) #绘制动画
ani.save('./record/m1_eval_3a_animation.gif') #显示动画

print('Trajectory Animation created.')
