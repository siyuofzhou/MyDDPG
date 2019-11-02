# -*- coding: utf-8 -*-
import numpy as np
from Parameter import P

# dx = theta*(mu-x)*dt + sigam*sqrt(dt)*Wt
# Wt是维纳过程
class UONoise:
    def __init__(self,size=4):
        self.theta = 0.15
        self.sigam = 0.2
        self.dt = 0.01
        self.mu = np.random.uniform(low=-0.5,high=0.5,size=size)
        
    def __call__(self,x,p=P['noise_p']):
        if np.random.uniform() > p:
            return x
        #x = self.mu
        x = x + self.theta*(self.mu-x)*self.dt+\
            self.sigam*np.sqrt(self.dt)*np.random.normal(size=x.shape)
        
        self.mu = np.random.uniform(low=-0.5,high=0.5,size=x.shape)
        return np.clip(x,-1.0,1.0)
    
