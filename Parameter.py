# -*- coding: utf-8 -*-
P = {
'TF_Log':'.\\log_v2',#tensorboard日志文件
'save_dir':'.\\model_v2',#模型文件夹路径
'save_path':'.\\model_v2\\model',#模型保存路劲
'save_file':'.\\model_v2\\checkpoint',#模型checkpoint文件路径
'memory_save_path':'.\\memory_v2',#记忆库保存路径
'batch_size':128,#每次训练的批大小
'train_one_eps':20,#一个eps训练的次数
'train_all_eps':100000,#总共运行的eps的次数
'memory_updata_size':1000,#每个eps记忆库更新记录数量
'policy_learn_rate':1e-3,#policy网络学习率
'Q_learn_rate':1e-3,#Q-value网络学习率
'gama':0.99,#折扣回馈法回馈率
'soft_update_rate':0.0001,#target网络更新比例
'l2_regularizer':1e-5,#正则化系数
'memory_upline':1000000,#记忆库上限
'policy_clip_up':2.0,#策略网络权重上限
'Q_clip_up':2.0,#Q-value网络权重上限
'mode':1,#0表示训练，1表示测试
'noise_p':0.1#UOnoise中进行噪声变幻的概率
}

