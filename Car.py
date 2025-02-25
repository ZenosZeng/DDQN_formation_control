import numpy as np
from math import cos,sin

class Car:
	def __init__(self,M,R,L,J,x=0.0,y=0.0,theta=0.0,vx=0.0,vy=0.0,vtheta=0.0,T_control=0.1):
		self.x = x
		self.y = y
		self.theta = theta
		self.vx = vx
		self.vy = vy
		self.vtheta = vtheta
		self.delta = 0
		self.T_control = T_control
		self.dt = self.T_control/10
		self.R = R
		self.M = M
		self.L = L
		self.J = J
		
	def reset(self,x,y,theta):
		self.x = x
		self.y = y
		self.theta = theta

	def step(self,ur,ul):
		# Dynamic model
		# ur ul -> ax ay atheta
		# solve a_theta
		atheta = (ur-ul)* self.L / self.J / self.R

		# solve ax ay AX=B
		A = np.array([[self.M*self.R*cos(self.theta),self.M*self.R*sin(self.theta)],
	                   [sin(self.theta),-cos(self.theta)]] )
		B = np.array( [[self.M*self.R*sin(self.theta)*self.vtheta*self.vx-self.M*self.R*cos(self.theta)*self.vtheta*self.vy\
		 				+ur+ul],
	                    [-cos(self.theta)*self.vtheta*self.vx-sin(self.theta)*self.vtheta*self.vy]] )
		solution = np.linalg.inv(A)*B
		# print(solution)
		ax = solution[0][0]
		ay = solution[1][0]
		# print(ax,ay)

		# kinematic model
		# vx vy vtheta -> x y theta
		self.x += self.vx*self.dt
		self.y += self.vy*self.dt
		self.theta += self.vtheta*self.dt

		# ax ay atheta -> vx vy vtheta
		self.vx += ax*self.dt
		self.vy += ay*self.dt
		self.vtheta += atheta*self.dt
		print(ax,ay)

print('Done')

