# all evaluation intergration
# Mission 3 test
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# ****************************************************************************************************************
# load model
# ****************************************************************************************************************

import torch
model_file_name= 'm2_DDQN_net'
model = torch.load(model_file_name)
model = model.to('cpu')

print('model Loaded.')

# ****************************************************************************************************************
# action selection Func
# ****************************************************************************************************************

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

    # Control of follower
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

# ****************************************************************************************************************
# cal state from 2 car
# ****************************************************************************************************************
def car2state(Leader,Follower,nof,com,formation):
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

# ****************************************************************************************************************
# Car ploting Func
# ****************************************************************************************************************
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

# ****************************************************************************************************************
# evaluation Env and Data Collector Class
# ****************************************************************************************************************

from CarNew import Car
from math import pi
from math import cos,sin
from theta_calibrate import theta_cali

from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import animation

from matplotlib import rcParams
rcParams['font.family'] = 'SimHei'
rcParams['axes.unicode_minus']=False

class ModelTest():

    '''
    coordinate:5 x y theta v w
    coor_list[i][j][k]
        i: coordinate from 0-4: x y theta v w
        j: Car No.
        k: time
    '''
    #-------------------------------------------------------------------------------------
    def __init__(self,TotalCar,TotalTime,TestName,StartPos,LeaderControl,Comlist,Formation,\
                 xpltrange=[-5,5],ypltrange=[-5,5],pltrange=[-20,20,-20,20],\
                 formation2=None,clicktime=0):
        
        # Para
        self.coordinate_total = 5
        self.car_total = TotalCar
        self.time_total = TotalTime
        self.name = TestName
        self.startpos = StartPos
        self.leadercontrol = LeaderControl
        self.com = Comlist
        self.formation = Formation
        self.f2 = formation2
        self.ct = clicktime
        self.pltrange = pltrange
        self.xrange = xpltrange
        self.yrange = ypltrange

        self.energy = 0

        # car and data list
        self.car=[ Car() for i in range(self.car_total) ]
        self.coor_list = [ [ [  ] for i in range(self.coordinate_total) ] for i in range(self.car_total) ]

        # reset all car
        for i in range(self.car_total):
            self.car[i].reset(StartPos[i][0],StartPos[i][1],StartPos[i][2],StartPos[i][3],StartPos[i][4])

        # add start coordinate
        for i in range(self.car_total):
            self.coor_list[i][0].append(self.car[i].x)
            self.coor_list[i][1].append(self.car[i].y)
            self.coor_list[i][2].append(self.car[i].theta)
            self.coor_list[i][3].append(self.car[i].v)
            self.coor_list[i][4].append(self.car[i].w)

        self.ex_list= [ [] for i in range(self.car_total)]
        self.ey_list= [ [] for i in range(self.car_total)]
            
        print('Env ready for {}.'.format(self.name))
        # print('All car initialized.')

    #-------------------------------------------------------------------------------------
    def getTra(self):
    # run the Env
        for t in range(self.time_total):

            if t>self.ct and self.f2 is not None:
                self.formation = self.f2

            ut = [ [0,0] for i in range(self.car_total)]
            #  leader
            ut[0]=[self.leadercontrol[t][0],self.leadercontrol[t][1]]

            # Follower
            for i in range(1,self.car_total):
                if(com[t][i]==-1):
                    ur=0
                    ul=0
                else:
                    state = car2state(self.car[self.com[t][i]],self.car[i],i,self.com[t],self.formation)
                    ur,ul = act(model,state)
                ut[i]=[ur,ul]

            # update all
            for i in range(self.car_total):
                self.car[i].run(ut[i][0],ut[i][1])
                if( i != 0 ):
                    self.energy += ( abs(ut[i][0])+abs(ut[i][1]) )*self.car[i].v*self.car[i].dt/self.car[i].R

            # save data of all car
            for i in range(self.car_total):
                self.coor_list[i][0].append(self.car[i].x)
                self.coor_list[i][1].append(self.car[i].y)
                self.coor_list[i][2].append(self.car[i].theta)
                self.coor_list[i][3].append(self.car[i].v)
                self.coor_list[i][4].append(self.car[i].w)

                self.ex_list[i].append(self.coor_list[0][0][-1] - self.coor_list[i][0][-1] + self.formation[i][0])
                self.ey_list[i].append(self.coor_list[0][1][-1] - self.coor_list[i][1][-1] + self.formation[i][1])

        # print(coor_list[0][0])
        # print('Run Completed & Data Collected.')
        print('Total energy:{:.4f}'.format(self.energy))
        return self.coor_list

    #-------------------------------------------------------------------------------------
    def Plot(self):
        color_list = ['red','green','limegreen','darkblue','blue']
        label_list = ['领导者','跟随者1','跟随者2','跟随者3','跟随者4']
        line_width = 0.5

        plt.figure(dpi=100)
        # trajectory for each car
        for i in range(self.car_total):
            plt.plot(self.coor_list[i][0],self.coor_list[i][1],color=color_list[i],label=label_list[i],linewidth=line_width)
            # plt.scatter(self.coor_list[i][0][0],self.coor_list[i][1][0],color=color_list[i],linewidth=line_width)
            plt.scatter(self.coor_list[i][0][0],self.coor_list[i][1][0],color=color_list[i],marker='x',linewidth=line_width) 
            x,y = draw_car(self.coor_list[i][0][-1],self.coor_list[i][1][-1],self.coor_list[i][2][-1])
            plt.plot(x,y,color=color_list[i],linewidth=line_width)
        # axe
        ax = plt.gca()
        ax.set_aspect(1)

        pltRange = self.pltrange

        plt.xlim(pltRange[0],pltRange[1])
        plt.ylim(pltRange[2],pltRange[3])
        plt.title('无人车轨迹')
        plt.xlabel('x/m')
        plt.ylabel('y/m')
        plt.legend()
        plt.savefig('./pic/{}_Tra.png'.format(self.name))

        # print('Trajectory Ploted.')

        #-------------------------------------------------------------------------------------
        # x error
        plt.figure(dpi=400)
        plt.subplot(211)
        # /
        # axes.spines['left'].set_position(('data', 0))

        plt.ylim(self.xrange[0],self.xrange[1]) # x error

        # plt.xlabel('t/s')
        plt.xticks([])
        plt.ylabel('x/m')
        for i in range(self.car_total):
            plt.plot(np.arange(0,self.time_total/10,0.1),self.ex_list[i],color=color_list[i],label=label_list[i])
        plt.title('x方向误差')
        plt.legend(loc=1,fontsize='xx-small') # 

        # y
        plt.subplot(212)
        axes = plt.gca()
        # axes.spines['right'].set_color('none')
        # axes.spines['top'].set_color('none')
        # axes.spines['bottom'].set_position(('data', 0))
        # axes.spines['left'].set_position(('data', 0))

        plt.ylim(self.yrange[0],self.yrange[1]) # y error

        plt.xlabel('t/s')
        plt.ylabel('y/m')
        for i in range(self.car_total):
            plt.plot(np.arange(0,self.time_total/10,0.1),self.ey_list[i],color=color_list[i],label=label_list[i])

        plt.title('y方向误差')
        # plt.legend()
        plt.savefig('./pic/{}_XYerror.png'.format(self.name))
        # finish
        # print('Error Ploted.')

        #-------------------------------------------------------------------------------------
        # animation

        fig = plt.figure(dpi=100)

        ax = fig.subplots()
        ax.set_aspect(1)
        ax.set_xlabel('x/m')
        ax.set_ylabel('y/m')
        ax.legend()

        line=[ 0 for i in range(2*self.car_total) ]

        t=0
        for i in range(self.car_total):
            line[i]=ax.plot(self.coor_list[i][0][:t],self.coor_list[i][1][:t],color=color_list[i],label=label_list[i],linewidth=line_width)
            line[i+self.car_total]=ax.plot(self.coor_list[0][0][:t],self.coor_list[0][1][:t],color=color_list[i],linewidth=line_width)    

        def update(t): 
            # pltRange = [-5,13,-9,9]
            ax.set_xlim(pltRange[0],pltRange[1])
            ax.set_ylim(pltRange[2],pltRange[3])

            ax.legend()
            ax.set_title('无人车轨迹 {:.1f}s'.format(t/10))
            for i in range(self.car_total):
                x,y = draw_car(self.coor_list[i][0][t],self.coor_list[i][1][t],self.coor_list[i][2][t])
                line[i][0].set_data(x,y)
                line[i+self.car_total][0].set_data(self.coor_list[i][0][:t],self.coor_list[i][1][:t])

            # print(line)
            return line

        ddt = 10
        ani=FuncAnimation(fig,update,interval=ddt,frames=self.time_total-1) 
        ani.save('./pic/{}_animation.gif'.format(self.name))

        print('Trajectory Animation create.')
        print('*'*50)
                
# ****************************************************************************************************************
# test activation
# ****************************************************************************************************************

T1A = 0 # 一致性 并列起步
T1B = 0 # 一致性 树形起步
T1C = 0 # 一致性 串联起步
T1D = 0 # 一致性 领导者训练轨迹
T1E = 1 # 一致性 领导者过山车轨迹

T2A = 0 # 五边形编队 并列起步
T2B = 0 # 五边形编队 树形起步
T2C = 0 # 五边形编队 串联起步
T2D = 0 # 五边形编队 领导者训练轨迹
T2E = 1 # 五边形编队 领导者过山车轨迹
T2F = 0 # 平行编队 并列起步
T2G = 0 # 方形编队 并列起步

T3A = 0 # 通讯切换
T3B = 0 # 通讯失效并恢复
T3C = 0 # 编队切换


'''

formation = [ [  0.0, 0.0 , 0.0 ],
                [  df*cos(2/5*pi)-df , df*sin(2/5*pi) , 0.0 ],
                [  df*cos(4/5*pi)-df , df*sin(4/5*pi) , 0.0 ],
                [  df*cos(8/5*pi)-df , df*sin(8/5*pi) , 0.0 ],
                [  df*cos(6/5*pi)-df , df*sin(6/5*pi) , 0.0 ]  ]

for t in range(time):
    if 150<t<170:
        leaderU[t]=(-0.2*0.001,0.2*0.001)
    elif 190<t<210:
        leaderU[t]=(0.2*0.001,-0.2*0.001)
    elif 210<t<230:
        leaderU[t]=(0.01,0.01)    
    elif 250<t<270:
        leaderU[t]=(-0.01,-0.01)
    elif 270<t<290:
        leaderU[t]=(0.2*0.001,-0.2*0.001)
    elif 310<t<330:
        leaderU[t]=(-0.2*0.001,0.2*0.001)
    else:
        pass

'''

# ****************************************************************************************************************
# T1 一致性
# ****************************************************************************************************************
# scene 1A 
if( T1A == 1 ):
    time = 300
    startpos = [   [0,0,0,0.2,0],
                    [0,2,0,0,0],
                    [0,4,0,0,0],
                    [0,-2,0,0,0],
                    [0,-4,0,0,0]    ]
    df = 1
    formation = [ [  0.0, 0.0 , 0.0 ],
                  [  0.0, 0.0 , 0.0 ],
                  [  0.0, 0.0 , 0.0 ],
                  [  0.0, 0.0 , 0.0 ],
                  [  0.0, 0.0 , 0.0 ]  ]

    com = [ (0,0,1,0,3) for t in range(time) ]

    leaderU = [ (0,0) for t in range(time) ]
    
    testscene = ModelTest(5,time,'m2_1A',startpos,leaderU,com,formation,[-2,2],[-5,5],[-2,10,-6,6])
    testscene.getTra()
    testscene.Plot()

# ****************************************************************************************************************
# scene 1B 
if( T1B == 1 ):
    time = 300
    startpos = [   [0,0,0,0.2,0],
                    [-1,2,0,0,0],
                    [-2,4,0,0,0],
                    [-1,-2,0,0,0],
                    [-2,-4,0,0,0]    ]
    df = 1
    formation = [ [  0.0, 0.0 , 0.0 ],
                  [  0.0, 0.0 , 0.0 ],
                  [  0.0, 0.0 , 0.0 ],
                  [  0.0, 0.0 , 0.0 ],
                  [  0.0, 0.0 , 0.0 ]  ]

    com = [ (0,0,1,0,3) for t in range(time) ]

    leaderU = [ (0,0) for t in range(time) ]
    
    testscene = ModelTest(5,time,'m2_1B',startpos,leaderU,com,formation,[-5,5],[-5,5],[-4,8,-6,6])
    testscene.getTra()
    testscene.Plot()

# ****************************************************************************************************************
# scene 1C
if( T1C == 1 ):
    time = 300
    startpos = [   [0,0,0,0.2,0],
                    [-1,0,0,0,0],
                    [-2,0,0,0,0],
                    [-3,0,0,0,0],
                    [-4,0,0,0,0]    ]
    df = 1
    formation = [ [  0.0, 0.0 , 0.0 ],
                  [  0.0, 0.0 , 0.0 ],
                  [  0.0, 0.0 , 0.0 ],
                  [  0.0, 0.0 , 0.0 ],
                  [  0.0, 0.0 , 0.0 ]  ]

    com = [ (0,0,1,0,3) for t in range(time) ]

    leaderU = [ (0,0) for t in range(time) ]
    
    testscene = ModelTest(5,time,'m2_1C',startpos,leaderU,com,formation,[-5,5],[-2,2],[-6,8,-7,7])
    testscene.getTra()
    testscene.Plot()

# ****************************************************************************************************************
# scene 1D 
if( T1D == 1 ):
    time = 400
    startpos = [   [0,0,0,0.2,0],
                    [0,2,0,0,0],
                    [0,4,0,0,0],
                    [0,-2,0,0,0],
                    [0,-4,0,0,0]    ]
    df = 1
    formation = [ [  0.0, 0.0 , 0.0 ],
                  [  0.0, 0.0 , 0.0 ],
                  [  0.0, 0.0 , 0.0 ],
                  [  0.0, 0.0 , 0.0 ],
                  [  0.0, 0.0 , 0.0 ]  ]

    com = [ (0,0,1,0,3) for t in range(time) ]

    leaderU = [ (0,0) for t in range(time) ]
    for t in range(time):
        if 150<t<170:
            leaderU[t]=(-0.2*0.001,0.2*0.001)
        elif 190<t<210:
            leaderU[t]=(0.2*0.001,-0.2*0.001)
        elif 210<t<230:
            leaderU[t]=(0.3*0.01,0.3*0.01)    
        elif 250<t<270:
            leaderU[t]=(-0.3*0.01,-0.3*0.01)
        elif 270<t<290:
            leaderU[t]=(0.2*0.001,-0.2*0.001)
        elif 310<t<330:
            leaderU[t]=(-0.2*0.001,0.2*0.001)
        else:
            pass
    
    testscene = ModelTest(5,time,'m2_1D',startpos,leaderU,com,formation,[-2,2],[-5,5],[-2,12,-7,7])
    testscene.getTra()
    testscene.Plot()

# ****************************************************************************************************************
# scene 1E 
if( T1E == 1 ):
    time = 1100
    startpos = [   [0,0,0,0.2,0],
                    [0,2,0,0,0],
                    [0,4,0,0,0],
                    [0,-2,0,0,0],
                    [0,-4,0,0,0]    ]
    df = 1
    formation = [ [  0.0, 0.0 , 0.0 ],
                  [  0.0, 0.0 , 0.0 ],
                  [  0.0, 0.0 , 0.0 ],
                  [  0.0, 0.0 , 0.0 ],
                  [  0.0, 0.0 , 0.0 ]  ]

    com = [ (0,0,1,0,3) for t in range(time) ]

    leaderU = [ (0,0) for t in range(time) ]
    for t in range(time):
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
    
    testscene = ModelTest(5,time,'m2_1E',startpos,leaderU,com,formation,[-2,2],[-5,5],[-2,10,-6,6])
    testscene.getTra()
    testscene.Plot()




# ****************************************************************************************************************
# T2 编队
# ****************************************************************************************************************
# scene 2A 
if( T2A == 1 ):
    time = 400
    startpos = [   [0,0,0,0.2,0],
                    [0,2,0,0,0],
                    [0,4,0,0,0],
                    [0,-2,0,0,0],
                    [0,-4,0,0,0]    ]
    df = 0.8
    formation = [ [  0.0, 0.0 , 0.0 ],
                  [  df*cos(2/5*pi)-df , df*sin(2/5*pi) , 0.0 ],
                  [  df*cos(4/5*pi)-df , df*sin(4/5*pi) , 0.0 ],
                  [  df*cos(8/5*pi)-df , df*sin(8/5*pi) , 0.0 ],
                  [  df*cos(6/5*pi)-df , df*sin(6/5*pi) , 0.0 ]  ]

    com = [ (0,0,1,0,3) for t in range(time) ]
    leaderU = [ (0,0) for t in range(time) ]

    testscene = ModelTest(5,time,'m2_2A',startpos,leaderU,com,formation,[-4,4],[-5,5],[-2,10,-6,6])
    testscene.getTra()
    testscene.Plot()

# ****************************************************************************************************************
# scene 2B 
if( T2B == 1 ):
    time = 400
    startpos = [   [0,0,0,0.2,0],
                    [-2,2,0,0,0],
                    [-4,4,0,0,0],
                    [-2,-2,0,0,0],
                    [-4,-4,0,0,0]    ]
    df = 0.8
    formation = [ [  0.0, 0.0 , 0.0 ],
                  [  df*cos(2/5*pi)-df , df*sin(2/5*pi) , 0.0 ],
                  [  df*cos(4/5*pi)-df , df*sin(4/5*pi) , 0.0 ],
                  [  df*cos(8/5*pi)-df , df*sin(8/5*pi) , 0.0 ],
                  [  df*cos(6/5*pi)-df , df*sin(6/5*pi) , 0.0 ]  ]

    com = [ (0,0,1,0,3) for t in range(time) ]
    leaderU = [ (0,0) for t in range(time) ]

    testscene = ModelTest(5,time,'m2_2B',startpos,leaderU,com,formation,[-5,5],[-5,5],[-5,9,-7,7])
    testscene.getTra()
    testscene.Plot()

# ****************************************************************************************************************
# scene 2C 
if( T2C == 1 ):
    time = 400
    startpos = [   [0,0,0,0.2,0],
                    [-1,0,0,0,0],
                    [-2,0,0,0,0],
                    [-3,0,0,0,0],
                    [-4,0,0,0,0]    ]
    df = 0.8
    formation = [ [  0.0, 0.0 , 0.0 ],
                  [  df*cos(2/5*pi)-df , df*sin(2/5*pi) , 0.0 ],
                  [  df*cos(4/5*pi)-df , df*sin(4/5*pi) , 0.0 ],
                  [  df*cos(8/5*pi)-df , df*sin(8/5*pi) , 0.0 ],
                  [  df*cos(6/5*pi)-df , df*sin(6/5*pi) , 0.0 ]  ]

    com = [ (0,0,1,0,3) for t in range(time) ]
    leaderU = [ (0,0) for t in range(time) ]

    testscene = ModelTest(5,time,'m2_2C',startpos,leaderU,com,formation,[-5,5],[-2,2],[-5,9,-7,7])
    testscene.getTra()
    testscene.Plot()

# ****************************************************************************************************************
# scene 2D 
if( T2D == 1 ):
    time = 400
    startpos = [   [0,0,0,0.2,0],
                    [0,2,0,0,0],
                    [0,4,0,0,0],
                    [0,-2,0,0,0],
                    [0,-4,0,0,0]    ]
    df = 0.8
    formation = [ [  0.0, 0.0 , 0.0 ],
                  [  df*cos(2/5*pi)-df , df*sin(2/5*pi) , 0.0 ],
                  [  df*cos(4/5*pi)-df , df*sin(4/5*pi) , 0.0 ],
                  [  df*cos(8/5*pi)-df , df*sin(8/5*pi) , 0.0 ],
                  [  df*cos(6/5*pi)-df , df*sin(6/5*pi) , 0.0 ]  ]

    com = [ (0,0,1,0,3) for t in range(time) ]
    leaderU = [ (0,0) for t in range(time) ]
    for t in range(time):
        if 150<t<170:
            leaderU[t]=(-0.2*0.001,0.2*0.001)
        elif 190<t<210:
            leaderU[t]=(0.2*0.001,-0.2*0.001)
        elif 210<t<230:
            leaderU[t]=(0.3*0.01,0.3*0.01)    
        elif 250<t<270:
            leaderU[t]=(-0.3*0.01,-0.3*0.01)
        elif 270<t<290:
            leaderU[t]=(0.2*0.001,-0.2*0.001)
        elif 310<t<330:
            leaderU[t]=(-0.2*0.001,0.2*0.001)
        else:
            pass

    testscene = ModelTest(5,time,'m2_2D',startpos,leaderU,com,formation,[-4,4],[-5,5],[-2,10,-6,6])
    testscene.getTra()
    testscene.Plot()

# ****************************************************************************************************************
# scene 2E 
if( T2E == 1 ):
    time = 1200
    startpos = [   [0,0,0,0.2,0],
                    [0,2,0,0,0],
                    [0,4,0,0,0],
                    [0,-2,0,0,0],
                    [0,-4,0,0,0]    ]
    df = 0.8
    formation = [ [  0.0, 0.0 , 0.0 ],
                  [  df*cos(2/5*pi)-df , df*sin(2/5*pi) , 0.0 ],
                  [  df*cos(4/5*pi)-df , df*sin(4/5*pi) , 0.0 ],
                  [  df*cos(8/5*pi)-df , df*sin(8/5*pi) , 0.0 ],
                  [  df*cos(6/5*pi)-df , df*sin(6/5*pi) , 0.0 ]  ]

    com = [ (0,0,1,0,3) for t in range(time) ]

    leaderU = [ (0,0) for t in range(time) ]
    for t in range(time):
        if 210<t<220:
            leaderU[t]=(-0.4*0.001,0.4*0.001)
        elif 560<t<570:
            leaderU[t]=(0.4*0.001,-0.4*0.001)
        if 750<t<760:
            leaderU[t]=(0.4*0.001,-0.4*0.001)
        elif 1100<t<1110:
            leaderU[t]=(-0.4*0.001,0.4*0.001)
        else:
            pass
    
    testscene = ModelTest(5,time,'m2_2E',startpos,leaderU,com,formation,[-4,4],[-5,5],[-2,12,-7,7])
    testscene.getTra()
    testscene.Plot()

# ****************************************************************************************************************
# scene 2F 
if( T2F == 1 ):
    time = 400
    startpos = [   [0,0,0,0.2,0],
                    [0,2,0,0,0],
                    [0,4,0,0,0],
                    [0,-2,0,0,0],
                    [0,-4,0,0,0]    ]
    # df = 1.2
    formation = [ [  0.0, 0.0 , 0.0 ],
                  [  0 , 1 , 0.0 ],
                  [  0 , 2 , 0.0 ],
                  [  0 , -1 , 0.0 ],
                  [  0 , -2 , 0.0 ]  ]

    com = [ (0,0,1,0,3) for t in range(time) ]
    leaderU = [ (0,0) for t in range(time) ]
    for t in range(time):
        if 150<t<170:
            leaderU[t]=(-0.2*0.001,0.2*0.001)
        elif 190<t<210:
            leaderU[t]=(0.2*0.001,-0.2*0.001)
        elif 210<t<230:
            leaderU[t]=(0.3*0.01,0.3*0.01)    
        elif 250<t<270:
            leaderU[t]=(-0.3*0.01,-0.3*0.01)
        elif 270<t<290:
            leaderU[t]=(0.2*0.001,-0.2*0.001)
        elif 310<t<330:
            leaderU[t]=(-0.2*0.001,0.2*0.001)
        else:
            pass

    testscene = ModelTest(5,time,'m2_2F',startpos,leaderU,com,formation,[-2,2],[-3,3],[-2,12,-7,7])
    testscene.getTra()
    testscene.Plot()

# ****************************************************************************************************************
# scene 2G 
if( T2G == 1 ):
    time = 400
    startpos = [   [0,0,0,0.2,0],
                    [0,2,0,0,0],
                    [0,4,0,0,0],
                    [0,-2,0,0,0],
                    [0,-4,0,0,0]    ]
    # df = 0.8
    formation = [ [  0.0, 0.0 , 0.0 ],
                  [  1 , 1 , 0.0 ],
                  [  -1 , 1 , 0.0 ],
                  [  1 , -1 , 0.0 ],
                  [  -1 , -1 , 0.0 ]  ]

    com = [ (0,0,1,0,3) for t in range(time) ]
    leaderU = [ (0,0) for t in range(time) ]
    for t in range(time):
        if 150<t<170:
            leaderU[t]=(-0.2*0.001,0.2*0.001)
        elif 190<t<210:
            leaderU[t]=(0.2*0.001,-0.2*0.001)
        elif 210<t<230:
            leaderU[t]=(0.3*0.01,0.3*0.01)    
        elif 250<t<270:
            leaderU[t]=(-0.3*0.01,-0.3*0.01)
        elif 270<t<290:
            leaderU[t]=(0.2*0.001,-0.2*0.001)
        elif 310<t<330:
            leaderU[t]=(-0.2*0.001,0.2*0.001)
        else:
            pass

    testscene = ModelTest(5,time,'m2_2G',startpos,leaderU,com,formation,[-2,2],[-5,5],[-2,12,-7,7])
    testscene.getTra()
    testscene.Plot()


# ****************************************************************************************************************
# T3 编队&special case
# ****************************************************************************************************************
# scene 3A 
if( T3A == 1 ):
    time = 400
    startpos = [   [0,0,0,0.2,0],
                    [0,2,0,0,0],
                    [0,4,0,0,0],
                    [0,-2,0,0,0],
                    [0,-4,0,0,0]    ]
    df = 0.8
    formation = [ [  0.0, 0.0 , 0.0 ],
                  [  df*cos(2/5*pi)-df , df*sin(2/5*pi) , 0.0 ],
                  [  df*cos(4/5*pi)-df , df*sin(4/5*pi) , 0.0 ],
                  [  df*cos(8/5*pi)-df , df*sin(8/5*pi) , 0.0 ],
                  [  df*cos(6/5*pi)-df , df*sin(6/5*pi) , 0.0 ]  ]

    com = [ (0,0,1,0,3) for t in range(time) ]
    for t in range(time):
        if 200<t<210:
            com[t]=(0,0,1,-1,3)
        elif t>210:
            com[t]=(0,0,1,1,3)

    leaderU = [ (0,0) for t in range(time) ]
    for t in range(time):
        if 150<t<170:
            leaderU[t]=(-0.2*0.001,0.2*0.001)
        elif 190<t<210:
            leaderU[t]=(0.2*0.001,-0.2*0.001)
        elif 210<t<230:
            leaderU[t]=(0.3*0.01,0.3*0.01)    
        elif 250<t<270:
            leaderU[t]=(-0.3*0.01,-0.3*0.01)
        elif 270<t<290:
            leaderU[t]=(0.2*0.001,-0.2*0.001)
        elif 310<t<330:
            leaderU[t]=(-0.2*0.001,0.2*0.001)
        else:
            pass

    testscene = ModelTest(5,time,'m2_3A',startpos,leaderU,com,formation,[-4,4],[-5,5],[-2,10,-6,6])
    testscene.getTra()
    testscene.Plot()

# ****************************************************************************************************************
# scene 3B 
if( T3B == 1 ):
    time = 400
    startpos = [   [0,0,0,0.2,0],
                    [0,2,0,0,0],
                    [0,4,0,0,0],
                    [0,-2,0,0,0],
                    [0,-4,0,0,0]    ]
    df = 0.8
    formation = [ [  0.0, 0.0 , 0.0 ],
                  [  df*cos(2/5*pi)-df , df*sin(2/5*pi) , 0.0 ],
                  [  df*cos(4/5*pi)-df , df*sin(4/5*pi) , 0.0 ],
                  [  df*cos(8/5*pi)-df , df*sin(8/5*pi) , 0.0 ],
                  [  df*cos(6/5*pi)-df , df*sin(6/5*pi) , 0.0 ]  ]

    com = [ (0,0,1,0,3) for t in range(time) ]
    for t in range(time):
        if 200<t<250:
            com[t]=(0,0,-1,0,3)

    leaderU = [ (0,0) for t in range(time) ]
    for t in range(time):
        if 150<t<170:
            leaderU[t]=(-0.2*0.001,0.2*0.001)
        elif 190<t<210:
            leaderU[t]=(0.2*0.001,-0.2*0.001)
        elif 210<t<230:
            leaderU[t]=(0.3*0.01,0.3*0.01)    
        elif 250<t<270:
            leaderU[t]=(-0.3*0.01,-0.3*0.01)
        elif 270<t<290:
            leaderU[t]=(0.2*0.001,-0.2*0.001)
        elif 310<t<330:
            leaderU[t]=(-0.2*0.001,0.2*0.001)
        else:
            pass

    testscene = ModelTest(5,time,'m2_3B',startpos,leaderU,com,formation,[-4,4],[-5,5],[-2,10,-6,6])
    testscene.getTra()
    testscene.Plot()

# ****************************************************************************************************************
# scene 3C 
if( T3C == 1 ):
    time = 400
    startpos = [   [0,0,0,0.2,0],
                    [0,2,0,0,0],
                    [0,4,0,0,0],
                    [0,-2,0,0,0],
                    [0,-4,0,0,0]    ]
    df = 0.8
    formation = [ [  0.0, 0.0 , 0.0 ],
                  [  df*cos(2/5*pi)-df , df*sin(2/5*pi) , 0.0 ],
                  [  df*cos(4/5*pi)-df , df*sin(4/5*pi) , 0.0 ],
                  [  df*cos(8/5*pi)-df , df*sin(8/5*pi) , 0.0 ],
                  [  df*cos(6/5*pi)-df , df*sin(6/5*pi) , 0.0 ]  ]
    formation2 = [ [  0.0, 0.0 , 0.0 ],
                  [  0 , 1 , 0.0 ],
                  [  0 , 2 , 0.0 ],
                  [  0 , -1 , 0.0 ],
                  [  0 , -2 , 0.0 ]  ]

    com = [ (0,0,1,0,3) for t in range(time) ]

    leaderU = [ (0,0) for t in range(time) ]
    for t in range(time):
        if 150<t<170:
            leaderU[t]=(-0.2*0.001,0.2*0.001)
        elif 190<t<210:
            leaderU[t]=(0.2*0.001,-0.2*0.001)
        elif 210<t<230:
            leaderU[t]=(0.3*0.01,0.3*0.01)    
        elif 250<t<270:
            leaderU[t]=(-0.3*0.01,-0.3*0.01)
        elif 270<t<290:
            leaderU[t]=(0.2*0.001,-0.2*0.001)
        elif 310<t<330:
            leaderU[t]=(-0.2*0.001,0.2*0.001)
        else:
            pass

    testscene = ModelTest(5,time,'m2_3C',startpos,leaderU,com,formation,[-3,3],[-5,5],[-2,10,-6,6],\
                          formation2=formation2,clicktime=250)
    testscene.getTra()
    testscene.Plot()
