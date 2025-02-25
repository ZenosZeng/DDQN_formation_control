# Mission 2 test 跟随者后移
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
#-------------------------------------------------------------------------------------
# DDQN pytorch model LOADING

import torch
model_file_name= 'm2_DDQN_net_finish'
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
coordinate_total = 5
car_total = 5
time_total = 400

car=[ Car() for i in range(car_total) ]
coor_list = [ [ [  ] for i in range(coordinate_total) ] for i in range(car_total) ]

print('Env ready.')

#-------------------------------------------------------------------------------------
# initization

# reset pos and velo
from math import pi
car[0].reset(0,0,0,v=0.2)
car[1].reset(-1,1.5,0)
car[2].reset(-2,3,0)
car[3].reset(-1,-1.5,0)
car[4].reset(-2,-3,0)

com = [0,0,1,0,3]

# Star formation
from math import pi,cos,sin
df = 1
formation = [ [  0.0, 0.0 , 0.0 ],
              [  df*cos(2/5*pi)-df , df*sin(2/5*pi) , 0.0 ],
              [  df*cos(4/5*pi)-df , df*sin(4/5*pi) , 0.0 ],
              [  df*cos(8/5*pi)-df , df*sin(8/5*pi) , 0.0 ],
              [  df*cos(6/5*pi)-df , df*sin(6/5*pi) , 0.0 ]  ]

# Trajectory ploting param
xmin=-3
xmax=13
ymin=-8
ymax=8
color_list = ['red','green','limegreen','darkblue','blue']
label_list = ['leader','follower1','follower2','follower3','follower4']
line_width = 0.8

for i in range(car_total):
    coor_list[i][0].append(car[i].x)
    coor_list[i][1].append(car[i].y)
    coor_list[i][2].append(car[i].theta)
    coor_list[i][3].append(car[i].v)
    coor_list[i][4].append(car[i].w)

print('All car initialized.')

#-------------------------------------------------------------------------------------
# run the car
from math import cos,sin

'''
Training State:
    theta_f = theta_cali(self.Follower.theta)
    theta_l = theta_cali(self.Leader.theta)
    ex =  self.Leader.x - self.Follower.x
    ey =  self.Leader.y - self.Follower.y
    e_theta = theta_l-theta_f
    e1 = ex*cos(theta_f)+ey*sin(theta_f)
    e2 = -ex*sin(theta_f)+ey*cos(theta_f)
    vf = self.Follower.v
    vl1 = self.Leader.v*cos(e_theta)
    vl2 = self.Leader.v*sin(e_theta)
    wf = self.Follower.w
    wl = self.Leader.w

    current_state = np.array([e1,e2,vf,vl1,vl2,wf,wl])
'''

from theta_calibrate import theta_cali
def car2state(Leader,Follower,nof):
    global formation,com

    nol = com[nof]
    xd = formation[nol][0]-formation[nof][0]
    yd = formation[nol][1]-formation[nof][1]
    thetad = formation[nol][2]-formation[nof][2]

    theta_f = theta_cali(Follower.theta)
    theta_l = theta_cali(Leader.theta)
    ex =  Leader.x - Follower.x - xd
    ey =  Leader.y - Follower.y - yd
    e_theta = theta_l-theta_f - thetad

    e1 = ex*cos(theta_f)+ey*sin(theta_f)
    e2 = -ex*sin(theta_f)+ey*cos(theta_f)
    vf = Follower.v
    vl1 = Leader.v*cos(e_theta)
    vl2 = Leader.v*sin(e_theta)
    wf = Follower.w
    wl = Leader.w

    current_state = np.array([e1,e2,vf,vl1,vl2,wf,wl])
    return current_state
    
# run the Env
for t in range(time_total):

    # update leader
    if t<100+50:
        car[0].run(0,0)
    elif 100+50<t<120+50:
        car[0].run(-0.2*0.001,0.2*0.001)
    elif 140+50<t<160+50:
        car[0].run(0.2*0.001,-0.2*0.001)
    elif 160+50<t<180+50:
        car[0].run(0.01,0.01)    
    elif 200+50<t<220+50:
        car[0].run(-0.01,-0.01)
    elif 220+50<t<240+50:
        car[0].run(0.2*0.001,-0.2*0.001)
    elif 260+50<t<280+50:
        car[0].run(-0.2*0.001,0.2*0.001)
    else:
        car[0].run(0,0)
    # if t<80+50:
    #     car[0].run(0,0)
    # elif 80+50<t<100+50:
    #     car[0].run(-0.5*0.001,0.5*0.001)
    # elif 100+50<t<120+50:
    #     car[0].run(0.5*0.001,-0.5*0.001)
    # elif 120+50<t<140+50:
    #     car[0].run(0.01,0.01)    
    # elif 140+50<t<160+50:
    #     car[0].run(-0.01,-0.01)
    # elif 160+50<t<180+50:
    #     car[0].run(0.5*0.001,-0.5*0.001)
    # elif 180+50<t<200+50:
    #     car[0].run(-0.5*0.001,0.5*0.001)
    # else:
    #     car[0].run(0,0)

    # update Follower
    for i in range(1,car_total):
        state = car2state(car[com[i]],car[i],i)
        ur,ul = act(model,state)
        car[i].run(ur,ul)

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

plt.figure(dpi=300)
# trajectory for each car
for i in range(car_total):
    plt.plot(coor_list[i][0],coor_list[i][1],color=color_list[i],label=label_list[i],linewidth=line_width)
    plt.scatter(coor_list[i][0][0],coor_list[i][1][0],color=color_list[i],linewidth=line_width)
    plt.scatter(coor_list[i][0][-1],coor_list[i][1][-1],color=color_list[i],marker='x',linewidth=line_width) 
# axe
ax = plt.gca()
ax.set_aspect(1)

plt.xlim(xmin,xmax)
plt.ylim(ymin,ymax)
plt.title('Car_Trajectory')
plt.xlabel('x/m')
plt.ylabel('y/m')
plt.legend()
plt.savefig('./record/m202b_trajectory.png')

print('Trajectory Ploted.')

#-------------------------------------------------------------------------------------
# Error Ploting

# x 
plt.figure(dpi=300)
plt.subplot(211)
axes = plt.gca()
axes.spines['right'].set_color('none')
axes.spines['top'].set_color('none')
axes.spines['bottom'].set_position(('data', 0))
# axes.spines['left'].set_position(('data', 0))
ex_list= [ [] for i in range(car_total)]
for i in range(1,car_total):
    ex_list[i] = [ coor_list[0][0][j] - coor_list[i][0][j] + formation[i][0] for j in range(time_total) ]

plt.ylim(-4,4) # x error

plt.xlabel('t/s')
plt.ylabel('x/m')
for i in range(1,car_total):
    plt.plot(np.arange(0,time_total/10,0.1),ex_list[i],color=color_list[i],label=label_list[i],linewidth=line_width)
plt.title('x_error')

# y
plt.subplot(212)
axes = plt.gca()
axes.spines['right'].set_color('none')
axes.spines['top'].set_color('none')
axes.spines['bottom'].set_position(('data', 0))
# axes.spines['left'].set_position(('data', 0))
ey_list= [ [] for i in range(car_total)]
for i in range(1,car_total):
    ey_list[i] = [ coor_list[0][1][j] - coor_list[i][1][j] + formation[i][1] for j in range(time_total) ]

plt.ylim(-4,4) # y error

plt.xlabel('t/s')
plt.ylabel('y/m')
for i in range(1,car_total):
    plt.plot(np.arange(0,time_total/10,0.1),ey_list[i],color=color_list[i],label=label_list[i],linewidth=line_width)

plt.title('y_error')
# plt.legend()
plt.savefig('./record/m202b_XYerror.png')
# finish
print('Error Ploted.')

#-------------------------------------------------------------------------------------
# Trajectory Animation
from matplotlib.animation import FuncAnimation
from matplotlib import animation

fig = plt.figure(dpi=300)
ax = fig.subplots()

t=0
ax.set_aspect(1)
ax.set_title('Car_Trajectory')
ax.set_xlabel('x/m')
ax.set_ylabel('y/m')
ax.legend()
line=[ 0 for i in range(2*car_total) ]
for i in range(car_total):
    line[i]=ax.plot(coor_list[i][0][:t],coor_list[i][1][:t],color=color_list[i],label=label_list[i])
    line[i+car_total]=ax.plot(coor_list[0][0][:t],coor_list[0][1][:t],color=color_list[i],linewidth=line_width)    

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

def update(i):  
    global t,car_total
    #plt.ylim(xmin,xmax)
    #plt.xlim(ymin,ymax)    
    ax.set_xlim(xmin,xmax)
    ax.set_ylim(ymin,ymax)
    ax.legend()
    ax.set_title('Car_Trajectory {:.1f}s'.format(t/10))
    t+=1
    for i in range(car_total):
        x,y = draw_car(coor_list[i][0][t],coor_list[i][1][t],coor_list[i][2][t])
        line[i][0].set_data(x,y)
        line[i+car_total][0].set_data(coor_list[i][0][:t],coor_list[i][1][:t])

    # print(line)
    return line

ani=FuncAnimation(fig,update,interval=20,frames=time_total-1) 
ani.save('./record/m202b_animation.gif') 

print('Trajectory Animation created.')
