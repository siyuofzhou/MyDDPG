# -*- coding: utf-8 -*-
import tensorflow as tf
from Network import JointNetWork
from Network_v2 import JointNetWork_V2
from Memory import Memory
import gym
import numpy as np
from UONoise import UONoise
import time 
from walker import BipedalWalker,BipedalWalkerHardcore

class Train:
    
    def __init__(self):
        self.batch_size = 64
        self.train_one_eps = 20
        self.train_all_eps = 100000
        self.memory_updata_size = self.batch_size*20
        tf.reset_default_graph()
        self.network = JointNetWork(24,4)
        self.sess = tf.Session()
        self.writer = tf.summary.FileWriter('.\\log_v2',self.sess.graph)
        self.network.restore(self.sess)
        self.sess.graph.finalize()
        self.memory = Memory()
        #self.env = gym.make('BipedalWalker-v2')
        self.env = BipedalWalkerHardcore()
        self.UO_noise = UONoise()
        
    def online_train(self):
        for i in range(self.train_one_eps):
            x = self.memory.get(self.batch_size)
            #print('record.shape',x.shape)
            online_state = x[:,:24]
            action = x[:,24:28]
            R_input = x[:,28:29]
            target_state = x[:,29:53]
            #print(R_input.shape)
            self.sess.run(self.network.policyUpdate,
                            feed_dict={
                                self.network.online_state_input:online_state,
                                self.network.action:action,
                                self.network.con_training_q:False,
                                self.network.bn_training:True
                                })
            merged,_,step = self.sess.run([self.network.merged,self.network.Qupdate,self.network.global_step],
                            feed_dict={
                                self.network.online_state_input:online_state,
                                self.network.target_state_input:target_state,
                                self.network.R_input:R_input,
                                self.network.action:action,
                                self.network.con_training_q:True,
                                self.network.bn_training:True
                                })
            self.sess.run(self.network.target_update)
            
            if step%100 == 0:
                self.writer.add_summary(merged,global_step=step)
    
    
    def __action(self,state):
        action = self.sess.run(self.network.online_action,
                           feed_dict={
                                self.network.online_state_input:state,
                                self.network.con_training_q:False,
                                self.network.action:np.zeros([self.batch_size,4]),
                                self.network.bn_training:False
                            })
        #print('online_action: ',action)
        return self.UO_noise(action[0]) 
    
    def memory_updata(self,show=False):
        state = self.env.reset()
        #print(state)
        cont = 0
        total_reward = 0
        nums = 0
        while cont < self.memory_updata_size:
            state = np.expand_dims(np.array(state,dtype=np.float32),0)
            #print('state :',state)
            action = self.__action(state)
            #print('OU_action :',action)
            assert action.shape == (4,)
            
            next_state,R,done,inf = self.env.step(action)
            if R <= -99.999:R = -1.0
            total_reward += R
            if show:
                self.env.render()
                print('total reward:{:.2f}  R:{:.2f}'.format(total_reward,R))
                print('action : ',action)
            self.memory.push(state[0],action,R,next_state)
            state = next_state
            cont += 1
            if done:
                if show:
                    self.env.close()
                    break
                state = self.env.reset()
                nums += 1
        if nums == 0:nums = 1
        return total_reward/nums
            
    def train(self):
        up_cont = 10
        rewards = np.zeros([up_cont])
        cont = 0
        for i in range(self.train_all_eps):
            s1 = time.time()
            var_reward = self.memory_updata()
            rewards[cont%10] = var_reward
            s2 = time.time()
            self.online_train()
            s3 = time.time()
            print ('index {}: memory update time: {:.2f} train time:{:.2f} reword:{:.2f}'.format(i,s2-s1,s3-s2,np.mean(rewards)))
            if (i+1)%100 == 0:
                self.network.save(self.sess)
                self.memory.save()
            cont+=1
        
    def test(self):
        self.memory_updata(show = True)

if __name__ == '__main__':
    train = Train()
    train.train()
    #train.test()