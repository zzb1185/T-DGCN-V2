# -*- coding: utf-8 -*-
# import pickle as pkl
# import tensorflow.compat.v1 as tf
# tf.compat.v1.disable_eager_execution()

import tensorflow as tf
import pandas as pd
import numpy as np
import math
import os
import numpy.linalg as la
from input_data import preprocess_data
from tgcn import tgcnCell
from tdgcn import tdgcnCell
from gru import GRUCell
from load import load_data
from visualization import plot_result,plot_error
from sklearn.metrics import mean_squared_error,mean_absolute_error
#import matplotlib.pyplot as plt
import time
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

time_start = time.time()

###### Settings ######
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rat e.')
flags.DEFINE_integer('training_epoch', 3000, 'N umber of epochs to train.')
flags.DEFINE_integer('gru_units', 128, 'hidden units of gru.[8,16,32,64,128,144]')
flags.DEFINE_integer('seq_len',7, '  time length of inputs.[7,14,21]')
flags.DEFINE_integer('pre_len', 1, 'time length of prediction.[1,3,5,7,14]')
flags.DEFINE_float('train_rate', 0.8, 'rate of training set.')
flags.DEFINE_integer('batch_size', 32, 'batch size.')
flags.DEFINE_string('dataset', 'test',
                    'test')
flags.DEFINE_string('model_name', 'tdgcn', 'tdgcn or tgcn')
model_name = FLAGS.model_name
data_name = FLAGS.dataset
train_rate = FLAGS.train_rate
seq_len = FLAGS.seq_len
output_dim = pre_len = FLAGS.pre_len
batch_size = FLAGS.batch_size
lr = FLAGS.learning_rate
training_epoch = FLAGS.training_epoch
gru_units = FLAGS.gru_units

###### load data ######
data, adj = load_data(data_name)
time_len = data.shape[0]
num_nodes = data.shape[1]
data1 =np.mat(data,dtype=np.float32)

#### normalization
max_value = np.max(data1)
min_value = np.min(data1)
std_value = data1.std()
mean_value = data1.mean()
data1 = data1/std_value
trainX, trainY, testX, testY = preprocess_data(data1, time_len, train_rate, seq_len, pre_len)

totalbatch = int(trainX.shape[0]/batch_size)
training_data_count = len(trainX)

def TDGCN(_X, _weights, _biases, _adj, _dataname):
    """
    函数介绍（中文版）
    描述：T-DGCN主函数，用来调用cell函数搭建网络。
    输入：特征矩阵、权重矩阵、偏置、空间矩阵、数据集名称
    与T-GCN的区别：tdgcnCell() 增加了数据集名称的输入，便于后续的DTW相似度计算。各项与DTW计算有关的步骤请见tdgcnCell()函数

    Function introduction (English version)
    Description: T-DGCN main function, used to call cell function to build the network.
    Input: feature matrix, weight matrix, bias, space matrix, dataset name
    Difference with T-GCN: tdgcnCell() adds the input of dataset name to facilitate the subsequent DTW similarity calculation.
                            See tdgcnCell() function for each step related to DTW calculation
    """
    cell_1 = tdgcnCell(gru_units, _adj, num_nodes=num_nodes, dataname=_dataname)
    cell = tf.nn.rnn_cell.MultiRNNCell([cell_1], state_is_tuple=True)
    _X = tf.unstack(_X, axis=1)
    outputs, states = tf.nn.static_rnn(cell, _X, dtype=tf.float32)
    m = []
    for i in outputs:
        o = tf.reshape(i,shape=[-1,num_nodes,gru_units])
        o = tf.reshape(o,shape=[-1,gru_units])
        m.append(o)
    last_output = m[-1]
    output = tf.matmul(last_output, _weights['out']) + _biases['out']
    output = tf.reshape(output,shape=[-1,num_nodes,pre_len])
    output = tf.transpose(output, perm=[0,2,1])
    output = tf.reshape(output, shape=[-1,num_nodes])
    return output, m, states

def TGCN(_X, _weights, _biases, _adj):
    """
    函数介绍（中文版）
    描述：T-GCN主函数，用来调用cell函数搭建网络。
    输入：特征矩阵、权重矩阵、偏置、空间矩阵
    与T-DGCN的区别：tgcnCell（）中不支持输入数据名称，后续不会调用DTW函数进行计算。相应的，在卷积层中不会计算哈达玛积修正函数。

    Function introduction (English version)
    Description: T-GCN main function, which is used to call the cell function to build the network.
    Input: feature matrix, weight matrix, bias, space matrix
    Difference with T-DGCN:The input of data names is not supported in tgcnCell(),
                            and the DTW function will not be called subsequently for calculation.
                            Correspondingly, the Hadamard product correction function will not be calculated in the convolution layer.
    """
    cell_1 = tgcnCell(gru_units, _adj, num_nodes=num_nodes)
    cell = tf.nn.rnn_cell.MultiRNNCell([cell_1], state_is_tuple=True)
    _X = tf.unstack(_X, axis=1)
    outputs, states = tf.nn.static_rnn(cell, _X, dtype=tf.float32)
    m = []
    for i in outputs:
        o = tf.reshape(i,shape=[-1,num_nodes,gru_units])
        o = tf.reshape(o,shape=[-1,gru_units])
        m.append(o)
    last_output = m[-1]
    output = tf.matmul(last_output, _weights['out']) + _biases['out']
    output = tf.reshape(output,shape=[-1,num_nodes,pre_len])
    output = tf.transpose(output, perm=[0,2,1])
    output = tf.reshape(output, shape=[-1,num_nodes])
    return output, m, states

def GRU(_X, _weights, _biases):
    ###
    cell_1 = GRUCell(gru_units, num_nodes=num_nodes)
    cell = tf.nn.rnn_cell.MultiRNNCell([cell_1], state_is_tuple=True)
    _X = tf.unstack(_X, axis=1)
    outputs, states = tf.nn.static_rnn(cell, _X, dtype=tf.float32)
    m = []
    for i in outputs:
        o = tf.reshape(i,shape=[-1,num_nodes,gru_units])
        o = tf.reshape(o,shape=[-1,gru_units])
        m.append(o)
    last_output = m[-1]
    output = tf.matmul(last_output, _weights['out']) + _biases['out']
    output = tf.reshape(output,shape=[-1,num_nodes,pre_len])
    output = tf.transpose(output, perm=[0,2,1])
    output = tf.reshape(output, shape=[-1,num_nodes])
    return output, m, states

###### placeholders占位符 ######
inputs = tf.compat.v1.placeholder(tf.float32, shape=[None, seq_len, num_nodes])
labels = tf.compat.v1.placeholder(tf.float32, shape=[None, pre_len, num_nodes])

# Graph weights
weights = {
    'out': tf.Variable(tf.random_normal([gru_units, pre_len], mean=1.0), name='weight_o')}
biases = {
    'out': tf.Variable(tf.random_normal([pre_len]),name='bias_o')}

if model_name == 'tdgcn':
    pred,ttts,ttto = TDGCN(inputs, weights, biases, adj, data_name)

if model_name == 'tgcn':
    pred,ttts,ttto = TGCN(inputs, weights, biases, adj)

y_pred = pred
      

###### optimizer ######
lambda_loss = 0.0015
# 正则化项
Lreg = lambda_loss * sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())
label = tf.reshape(labels, [-1,num_nodes])
##loss L2+L2
loss = tf.reduce_mean(tf.nn.l2_loss(y_pred-label) + Lreg)
##rmse
error = tf.sqrt(tf.reduce_mean(tf.square(y_pred-label)))
optimizer = tf.train.AdamOptimizer(lr).minimize(loss)

###### Initialize session ######
variables = tf.global_variables()
saver = tf.train.Saver(tf.global_variables())  
#sess = tf.Session()
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
sess.run(tf.global_variables_initializer())

out = 'out/%s'%(model_name)
#out = 'out/%s_%s'%(model_name,'perturbation')
time1 = time.time()
path1 = '%s_%s_lr%r_batch%r_unit%r_seq%r_pre%r_epoch%r_time%r'%(model_name,data_name,lr,batch_size,gru_units,seq_len,pre_len,training_epoch,time1)
path = os.path.join(out,path1)
if not os.path.exists(path):
    os.makedirs(path)
    
###### evaluation ######
def evaluation(a,b):
    rmse = math.sqrt(mean_squared_error(a,b))
    mae = mean_absolute_error(a, b)
    F_norm = la.norm(a-b,'fro')/la.norm(a,'fro')
    return rmse, mae, 1-F_norm
 
   
x_axe,batch_loss,batch_rmse,batch_pred = [], [], [], []
test_loss,test_rmse,test_mae,test_acc,test_r2,test_var,test_pred = [],[],[],[],[],[],[]
  
for epoch in range(training_epoch):
    for m in range(totalbatch):
        mini_batch = trainX[m * batch_size : (m+1) * batch_size]
        mini_label = trainY[m * batch_size : (m+1) * batch_size]
        _, loss1, rmse1, train_output = sess.run([optimizer, loss, error, y_pred],
                                                 feed_dict = {inputs:mini_batch, labels:mini_label})
        batch_loss.append(loss1)
        batch_rmse.append(rmse1*std_value )

     # Test completely at every epoch
    loss2, rmse2, test_output = sess.run([loss, error, y_pred],
                                         feed_dict = {inputs:testX, labels:testY})
    test_label = np.reshape(testY,[-1,num_nodes])
    #duizhao
    test_label1 = test_label* std_value
    #yucezhi
    test_output1 = test_output* std_value
    rmse, mae, acc = evaluation(test_label1, test_output1)
    test_loss.append(loss2)
    test_rmse.append(rmse)
    test_mae.append(mae)
    test_acc.append(acc)

    test_pred.append(test_output1)
    
    print('Iter:{}'.format(epoch),
          'train_rmse:{:.4}'.format(batch_rmse[-1]),
          'test_loss:{:.4}'.format(loss2),
          'test_rmse:{:.4}'.format(rmse),
          'test_mae:{:.4}'.format(mae),
          'test_acc:{:.4}'.format(acc),
          )
    
    if (epoch % 500 == 0):        
        saver.save(sess, path+'/model_100/TGCN_pre_%r'%epoch, global_step = epoch)
        
time_end = time.time()
print(time_end-time_start,'s')

############## visualization ###############
b = int(len(batch_rmse)/totalbatch)
batch_rmse1 = [i for i in batch_rmse]
train_rmse = [(sum(batch_rmse1[i*totalbatch:(i+1)*totalbatch])/totalbatch) for i in range(b)]
batch_loss1 = [i for i in batch_loss]
train_loss = [(sum(batch_loss1[i*totalbatch:(i+1)*totalbatch])/totalbatch) for i in range(b)]

index = test_rmse.index(np.min(test_rmse))
test_result = test_pred[index]
# var = pd.DataFrame(test_result)
# var.to_csv(path+'/test_result.csv',index = False,header = False)

plot_error(train_rmse,train_loss,test_rmse,test_acc,test_mae,path)

print('min_rmse:%r'%(np.min(test_rmse)),
      'min_mae:%r'%(test_mae[index]),
      'max_acc:%r'%(test_acc[index]))