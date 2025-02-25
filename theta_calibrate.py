from math import pi
from matplotlib import pyplot as plt
import numpy as np

def theta_cali(theta):
    if theta > pi:
        return theta_cali(theta-2*pi)
    elif theta < -pi:
        return theta_cali(theta+2*pi)
    else:
        return theta
    
def draw():
    plt.figure()
    thetalist = np.arange(-5*pi,5*pi,0.01)
    cali_list = [ theta_cali(x) for x in thetalist] 
    plt.plot(thetalist,cali_list)
    plt.show()

# draw()