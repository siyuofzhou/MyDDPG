# -*- coding: utf-8 -*-
import numpy as np
import os
from Parameter import P

class Memory:
    def __init__(self):
        self.upline = P['memory_upline']
        self.index = 0
        self.index_arr = np.random.randint(0,high=self.upline,size=self.upline)
        self.__save_path = P['memory_save_path']
        if os.path.isfile(self.__save_path+'.npz'):
            p = np.load(self.__save_path+'.npz')
            print ('load '+self.__save_path+'.npz')
            self.__data = p['data']
            self.data_size = p['size'][0]
        else:
            self.__data = np.zeros((self.upline,53))
            self.data_size  = 0
            
    def __next_index(self):
        x = self.index_arr[self.index]
        self.index += 1
        if self.index >= self.upline:
            self.index = 0
            self.index_arr = np.random.randint(0,high=self.upline,size=self.upline)
        return x
    
    def push(self,state,action,reward,next_state):
        x = np.concatenate((state,action,[reward],next_state))
        
        if self.upline>self.data_size:
            self.__data[self.data_size] = x
            self.data_size += 1
        else:
            self.__data[self.__next_index()] = x
        #print(self.__data.shape)
        
    def get(self,batch_size):
        a = np.random.randint(0,high = self.data_size,size=batch_size)
        return self.__data[a,:]
    
    def save(self):
        np.savez(self.__save_path,data = self.__data,size = np.array([self.data_size]))
    
    def pri(self):
        print(self.__data)
        print(self.__data.shape)
    
    
if __name__ == '__main__':
    m = Memory()
    m.push([1.0,2.0,np.pi],[1,2],55,[2,3,1])
    m.save()
    m.pri()