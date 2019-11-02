# -*- coding: utf-8 -*-
import tensorflow as tf
import os
from tensorflow.python.ops.control_flow_ops import switch,merge
from Parameter import P

class JointNetWork_V2:
    
    def __init__(self,state_dim,action_dim):
        
        self.__soft_update_rate = P['soft_update_rate'] #soft更新target中的变量的变换比例
        self.__state_dim = state_dim
        self.__action_dim = action_dim
        self.__gama = P['gama'] #折扣回馈法中的系数
        self.__policy_learn_rate = P['policy_learn_rate']
        self.__Q_learn_rate = P['Q_learn_rate']
        
        self.global_step = tf.Variable(0,name='global_step')
        self.online_state_input = tf.placeholder(tf.float32,shape=(None,state_dim),name='online_state_input')
        #self.online_action_input = tf.placeholder(tf.float32,shape=(-1,action_dim),name='online_action_input')
        self.target_state_input = tf.placeholder(tf.float32,shape=(None,state_dim),name='target_state_input')
        self.R_input = tf.placeholder(tf.float32,shape=(None,1),name='R_input')
        self.action = tf.placeholder(tf.float32,shape=(None,action_dim),name='online_action_output')
        self.con_training_q = tf.placeholder(tf.bool,shape=[],name='con_training_q')
        self.bn_training = tf.placeholder(tf.bool,shape=[],name='bn_training')
        
        self.__save_path = P['save_path']
        self.__save_dir = P['save_dir']
        self.__save_file = P['save_file']
        
        self.__build()
        self.__SAVER()
        
    def __dense(self,input,uint,collections=None,active=True,name=None,reg=None):
        with tf.variable_scope(name):
            w = tf.get_variable('w',shape=(input.shape[1],uint),initializer=tf.truncated_normal_initializer(stddev=0.01),collections=collections,regularizer=reg)
            b = tf.get_variable('b',shape=(uint,),initializer=tf.zeros_initializer(),collections=collections,regularizer=reg)
            c = tf.add(tf.matmul(input,w),b)
            if active:
                o = tf.nn.relu(c)
            else:
                o= c
            return o
    # struction of Policy-net is that all of layers are dense connection
    # the nodes of layers in order is [state_dim,64,128,64,action_dim]
    # struction of Q-net is that all of layers are dense collection
    # the nodes of layers in order is [state_dim+action_dim,64,128,64,1]
    # trainable various are in collocetion ''netname'+_train_vars'
    def __network(self,input,netname,online=True,reg=None,bn_training = False):
        collections = [tf.compat.v1.GraphKeys.GLOBAL_VARIABLES,netname+'_train_vars']
        
        with tf.variable_scope(netname,reuse=tf.AUTO_REUSE):  
            policyLayers = [self.__state_dim,64,128,256,128,64] 
            QLayers = [self.__state_dim+self.__action_dim,64,128,256,128,64] 
            with tf.variable_scope('policy',reuse=tf.AUTO_REUSE):
                i = input
                #i = tf.layers.batch_normalization(input,training=bn_training)
                for index,layer in enumerate(policyLayers):
                    i = self.__dense(i,layer,collections=collections,name='dense_'+str(index),reg=reg)
                    
                act_x = self.__dense(i,self.__action_dim,collections=collections,active=False,name='dense_action',reg=reg)
                with tf.variable_scope('act'):
                    act = tf.tanh(act_x)
            
            if online:      
                (sw_policy,_) = switch(act,self.con_training_q,name='sw_policy')
                (_,sw_action) = switch(self.action,self.con_training_q,name='sw_action')
                (act,_) = merge([sw_policy,sw_action],name='merge')
                
            with tf.variable_scope('Q',reuse=tf.AUTO_REUSE):
                i = tf.concat([act,input],1)
                #i = tf.layers.batch_normalization(i,training=bn_training)
                for index,layer in enumerate(QLayers):
                    i = self.__dense(i,layer,collections=collections,name='dense_'+str(index),reg=reg)
                i = self.__dense(i,1,collections=collections,active=False,name='q_value',reg=reg)
        
        if online:
            return [act_x,act,i]
        else:
            return i
    
    
    def __QLoss(self,y_true,y_pre,reg=None,name=None):
        with tf.variable_scope(name):
            return tf.reduce_mean(tf.square(y_true-y_pre))+reg
        
    def __Q__loss(self,y_true,y_pre,name=None):
        with tf.variable_scope(name):
            return tf.reduce_mean(tf.square(y_true-y_pre))
    
    def __PolicyLoss(self,y_true,act_x,reg=None,name=None):
        with tf.variable_scope(name):
            return -1.0*tf.reduce_mean(y_true) + reg + 1e-3*tf.reduce_mean(tf.square(act_x))
    
    def __var_list_to_dict(self,var_list):
        p = {}
        for i in var_list:
            p[i.name] = i
        return p
    
    def __target_update(self):
        self.__online_vars_by_name = self.__var_list_to_dict(tf.get_collection('online_train_vars')) 
        self.__target_vars_by_name = self.__var_list_to_dict(tf.get_collection('target_train_vars')) 
        update_ops = []
        for (online_name,online_var) in self.__online_vars_by_name.items():
            target_name = online_name.replace('online','target',1)
            target_var = self.__target_vars_by_name[target_name]
            soft_var = online_var*self.__soft_update_rate + target_var*(1-self.__soft_update_rate)
            op = tf.assign(target_var,soft_var)
            update_ops.append(op)
        return tf.group(update_ops)
        
    def __build(self):
        reg_l2 = tf.contrib.layers.l2_regularizer(P['l2_regularizer'])
        
        self.target_Q = self.__network(self.target_state_input,'target',online=False)
        act_x,self.online_action,self.online_Q = self.__network(self.online_state_input,'online',reg=reg_l2,bn_training=self.bn_training)
        
        policy_set_l2 = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES,scope='online/policy')
        Q_set_l2 = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES,scope='online/Q')
        
        Q_loss_l2 = tf.add_n(Q_set_l2)
        policy_loss_l2 = tf.add_n(policy_set_l2)
        
        with tf.variable_scope('Loss'):
            self.QLoss = self.__QLoss(self.online_Q,self.target_Q*self.__gama+self.R_input,reg=Q_loss_l2,name='Q')
            self.policyLoss = self.__PolicyLoss(self.online_Q,act_x,reg = policy_loss_l2,name='policy')
            self.PLoss = tf.reduce_mean(self.online_Q)
            Q_loss = self.__Q__loss(self.online_Q,self.target_Q*self.__gama+self.R_input,name='Q_loss')
            
        scalar_loss = tf.summary.scalar('loss',Q_loss)
        scalar_Q = tf.summary.scalar('policy_Q',self.PLoss)
        scalar_reg = tf.summary.scalar('reg',Q_loss_l2+policy_loss_l2)
        
        self.merged = tf.summary.merge([scalar_loss,scalar_Q,scalar_reg])
        #self.merged_Q = tf.summary.merge([scalar_Q])
        
        policy_var_list = tf.get_collection('online_train_vars',scope='online/policy')
        Q_var_list = tf.get_collection('online_train_vars',scope='online/Q')
        print(policy_var_list)
        print(Q_var_list)
        #self.Qupdate = tf.train.AdamOptimizer(self.__Q_learn_rate).minimize(self.QLoss,global_step=self.global_step,var_list=Q_var_list)
        bn_update_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(bn_update_op):
            optimizer = tf.train.AdamOptimizer(self.__Q_learn_rate)
            grads = optimizer.compute_gradients(self.QLoss,var_list=Q_var_list)
            with tf.name_scope('Q_clip'):
                for i,(g,v) in enumerate(grads):
                    if g is not None:
                        grads[i] = (tf.clip_by_value(g,-P['Q_clip_up'],P['Q_clip_up']),v)
            self.Qupdate = optimizer.apply_gradients(grads)
        
        #self.policyUpdate = tf.train.AdamOptimizer(self.__policy_learn_rate).minimize(self.policyLoss,global_step=self.global_step,var_list=policy_var_list)
            optimizer = tf.train.AdamOptimizer(self.__policy_learn_rate)
            grads = optimizer.compute_gradients(self.policyLoss,var_list=policy_var_list)
            with tf.name_scope('policy_clip'):
                for i,(g,v) in enumerate(grads):
                    if g is not None:
                        grads[i] = (tf.clip_by_value(g,-P['policy_clip_up'],P['policy_clip_up']),v)
            self.policyUpdate = optimizer.apply_gradients(grads)
        
        self.global_step = tf.assign_add(self.global_step,1)
        
        self.target_update = self.__target_update()
        
    def __SAVER(self):
        var_list = tf.trainable_variables()
        g_list = tf.global_variables()
        bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
        bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
        var_list += bn_moving_vars
        print ('\n\n\nvar_list.................\n\n\n')
        for var in var_list:
            print(var)
        print ('\n\n\ng_list.................\n\n\n')
        for var in g_list:
            print(var)
        self.__Saver = tf.train.Saver(var_list=var_list,max_to_keep=2)
    
    def save(self,sess):
        self.__Saver.save(sess,self.__save_path,global_step=self.global_step)
    
    def restore(self,sess):
        sess.run(tf.global_variables_initializer())
        if os.path.isfile(self.__save_file):
            print('restore')
            self.__Saver.restore(sess,tf.train.latest_checkpoint(self.__save_dir))
        
