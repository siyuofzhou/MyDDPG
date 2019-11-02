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
from Parameter import P

class Train:
    def __init__(self):
        self.batch_size = P['batch_size']
        self.train_one_eps = P['train_one_eps']
        self.train_all_eps = P['train_all_eps']
        self.memory_updata_size = P['memory_updata_size']
        tf.reset_default_graph()
        self.network = JointNetWork_V2(24,4)
        self.sess = tf.Session()
        self.writer = tf.summary.FileWriter(P['TF_Log'],self.sess.graph)
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
        if P['mode'] == 0:
            print(u'\r online_action: {}                        '.format(action),end='')
            return self.UO_noise(action[0]) 
        return action[0]
    
    def memory_updata(self):
        state = self.env.reset()
        cont = 0
        total_reward = 0
        nums = 0
        while cont < self.memory_updata_size:
            state = np.expand_dims(np.array(state,dtype=np.float32),0)
            action = self.__action(state)
            assert action.shape == (4,)
            
            next_state,R,done,inf = self.env.step(action)
            total_reward += R
            self.memory.push(state[0],action,R,next_state)
            state = next_state
            cont += 1
            if done:
                #self.env.seed(3130)
                state = self.env.reset()
                nums += 1
        if nums == 0:nums = 1
        return total_reward/nums
    
    def check(self,show=True):
        state = self.env.reset()
        total_reward = 0
        while True:
            state = np.expand_dims(np.array(state,dtype=np.float32),0)
            action = self.__action(state)
            assert action.shape == (4,)
            
            next_state,R,done,inf = self.env.step(action)
            total_reward += R
            if show:
                self.env.render()
            state = next_state
            if done:
                if show:
                    self.env.close()
                break
        return total_reward
    
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
        
    def test(self,times = 100):
        MinR = 10000
        Minseed = 0 
        for i in range(times):
            seed = np.random.randint(1,high=10000)
            #seed = 3130
            self.env.seed(seed = seed)
            print(seed)
            reward = self.check(show=True)
            print('reward:{} seed:{} '.format(reward,seed))
            if reward < MinR:
                MinR = reward
                Minseed = seed
        print('minreward:{} minseed:{}'.format(MinR,Minseed))

if __name__ == '__main__':
    train = Train()
    if P['mode'] == 0:
        train.train()
    else:
        train.test()
#minreward:12.21449920929181 minseed:5090
#minreward:5.478706493673222 minseed:3130