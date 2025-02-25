from typing import Union
from math import cos,sin,pi
import numpy as np

import gym
from gym import spaces
# from gym.envs.classic_control import rendering  

# custon package
from CarNew import Car
from theta_calibrate import theta_cali
import random

class CarFollow(gym.Env[np.ndarray, Union[int, np.ndarray]]):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self):
        self.action_space=spaces.Discrete(9)
        self.observation_space=spaces.Box(low=np.array([ -100.0 for x in range(10) ]),\
                                          high=np.array([ 100.0 for x in range(10) ]) )
        self.Leader = Car()
        self.Follower = Car()
        self.time = 0
        self.done = False
        self.flag = 0
        self.iex = 0.0
        self.iey= 0.0

    def step(self, action):

        # Control of follower
        self.time+=1
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

        # update the car
        self.Follower.run(ur,ul)
        # self.Leader.run(0,0)
        if self.time<80:
            self.Leader.run(0,0)
        elif 80<self.time<100:
            self.Leader.run(-0.5*0.001,0.5*0.001)
        elif 100<self.time<120:
            self.Leader.run(0.5*0.001,-0.5*0.001)
        elif 120<self.time<140:
            self.Leader.run(0.01,0.01)    
        elif 140<self.time<160:
            self.Leader.run(-0.01,-0.01)
        elif 160<self.time<180:
            self.Leader.run(0.5*0.001,-0.5*0.001)
        elif 180<self.time<200:
            self.Leader.run(-0.5*0.001,0.5*0.001)
        # elif 250<self.time<260:
        #     self.Leader.run(-0.5*0.001,0.5*0.001)
        # elif 260+270<self.time<270+270:
        #     self.Leader.run(0.5*0.001,-0.5*0.001)
        else:
            self.Leader.run(0,0)

        #-------------------------------------------------------------------------------------
        # Reward function and Termination Condition
        theta_f = theta_cali(self.Follower.theta)
        theta_l = theta_cali(self.Leader.theta)

        ex =  self.Leader.x - self.Follower.x
        ey =  self.Leader.y - self.Follower.y
        self.iex += ex/10
        self.iey += ey/10
        ed2 =  ( ex**2+ey**2 )
        ed = ed2**0.5
        # e1 = ex*cos(theta_f)+ey*sin(theta_f)
        # e2 = ey*cos(theta_f)-ex*sin(theta_f)
        # e_theta = theta_l-theta_f
        evx = self.Leader.v*cos(self.Leader.theta)-self.Follower.v*cos(self.Follower.theta)
        evy = self.Leader.v*sin(self.Leader.theta)-self.Follower.v*sin(self.Follower.theta)
        ev=(evx**2+evy**2)**0.5
        
        rd=0
        if ed>5: rd=0
        elif 1<ed<=5: rd= 2.5*(5-ed) # 0 10
        elif 0.1<ed<=1: rd=10+100*(1-ed) # 10 100
        elif 0.01<ed<=0.1: rd=100+1000*(0.1-ed) # 100 190
        elif ed<=0.01: rd=190 # 190
        rv = -10*ev

        reward = rd + rv # + rv + rth

        if self.time >= 300 or ed > 5 : # or abs(y_error)>1
            self.done = True
        else:
            self.done = False

        #-------------------------------------------------------------------------------------

        current_state = np.array([ex,ey,self.iex,self.iey,\
                                  self.Follower.v*cos(theta_f),self.Follower.v*sin(theta_f),self.Follower.w,\
                                  self.Leader.v*cos(theta_l),self.Leader.v*sin(theta_l),self.Leader.w])# 

        info = {'leader_state':[self.Leader.x,self.Leader.y,self.Leader.theta,self.Leader.v,self.Leader.w],
                'follower_state':[self.Follower.x,self.Follower.y,self.Follower.theta,self.Follower.v,self.Follower.w],
                'time':self.time }

        return current_state,reward,self.done,info


    def reset(self,randstart=True):
        # initial postion
        if randstart==True:
            thetaf_start = random.uniform(-pi,pi)
            df_start = random.uniform(0,2)
            fy = df_start*sin(thetaf_start)
            fx = df_start*cos(thetaf_start)

            l_theta = random.uniform(-pi/2,pi/2)
            f_theta = random.uniform(-pi/2,pi/2)
            self.Leader.reset(0,0,l_theta,v=0.2)
            self.Follower.reset(fx,fy,f_theta)
        else:
            self.Leader.reset(0,0,0,v=0.2)
            self.Follower.reset(0,-1,0)


        self.time=0
        self.done = False
        self.iex = 0
        self.iey = 0
        
        ex =  self.Leader.x - self.Follower.x
        ey =  self.Leader.y - self.Follower.y

        theta_f = theta_cali(self.Follower.theta)
        theta_l = theta_cali(self.Leader.theta)
        # e1 = ex*cos(theta_f)+ey*sin(theta_f)
        # e2 = ey*cos(theta_f)-ex*sin(theta_f)
        # evx = self.Leader.v*cos(self.Leader.theta)-self.Follower.v*cos(self.Follower.theta)
        # evy = self.Leader.v*sin(self.Leader.theta)-self.Follower.v*sin(self.Follower.theta)

        current_state = np.array([ex,ey,self.iex,self.iey,\
                                  self.Follower.v*cos(theta_f),self.Follower.v*sin(theta_f),self.Follower.w,\
                                  self.Leader.v*cos(theta_l),self.Leader.v*sin(theta_l),self.Leader.w])# 


        return current_state
   
    def render(self, mode='human', close=False):
        pass
        
    def close(self):
        pass
        