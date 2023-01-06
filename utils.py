# -*- coding: utf-8 -*-

import tensorflow as tf
import scipy.sparse as sp
import numpy as np
import pandas as pd
def tensor2array(tensor):
    # input:tensor
    # output:ndarray
    session1 = tf.Session()
    array1 = session1.run(tensor)
    return array1

def hadamard_polymerization(adj_spa, adj_geo): # adj  dtw
    """
    函数介绍（中文版）
    描述：计算哈达玛积
    输入：DTW计算后得到的地质特征相似性矩阵 空间临近矩阵
    输出：地质特征与空间特征聚合后得到的混合矩阵

    Function introduction (English version)
    Description: Calculate Hadamard product
    Input: similarity matrix of geological features obtained after DTW calculation Spatial proximity matrix
    Output: hybrid matrix obtained after aggregation of geological features and spatial features
    """
    # 稀疏矩阵转为稠密矩阵 sparse_tensor_to_dense
    tf1 = tf.sparse_tensor_to_dense(adj_spa,
                                    default_value=0,
                                    validate_indices=True,
                                    name=None)
    session = tf.Session()
    # dense tensor to array
    array = session.run(tf1)
    # array to df
    pd1 = pd.DataFrame(array)
    # geo_mat to df
    pd2 = pd.DataFrame(adj_geo)
    # 计算哈达姆积  Hadamard product
    hdm = pd1*pd2

    # normolize
    arr1 = np.array(hdm)

    # 转为tensor准备计算tanh
    t = tf.convert_to_tensor(arr1, tf.float32, name='t')

    # 使用tanh（）作为激活函数，防止多次计算后矩阵数值弥散的问题
    # Use tanh() as the activation function to prevent the problem of matrix value dispersion after multiple calculations
    tanh_tensor = tf.tanh(t)
    tanh_array = tensor2array(tanh_tensor)
    # 调用功能函数，将乘积矩阵进行一些冗长的数据格式转化,返回tuple+稀疏矩阵

    # array to df
    tanh_df = pd.DataFrame(tanh_array)
    # 返回tuple + 稀疏矩阵
    hdm = pd2spa_tuple(tanh_df)

    return hdm

    # arr_min = arr1.min()
    # arr_max = arr1.max()
    # arr2 = (arr1-arr_min)/(arr_max)
    # arr2[arr2==0] = -np.inf

    # # 数据格式转换，便于后续计算，并无实际意义
    # # Data format conversion is convenient for subsequent calculation and has no practical significance
    # adj_1 = sp.coo_matrix(adj_a)
    # adj_2 = np.mat(pd.DataFrame(adj_b))
    # adj_2 = sp.coo_matrix(adj_2)
    #
    # # 将为数据转为密集矩阵，准备下一步计算 Convert the data into a dense matrix and prepare for the next calculation
    # adj_1 = sp.csr_matrix(adj_1)
    # adj_1 = adj_1.astype(np.float32).todense()
    # adj_2 = sp.csr_matrix(adj_2)
    # adj_2 = adj_2.astype(np.float32).todense()
    #
    # # 计算哈达玛积 Calculate Hadamard product
    # hadamard_adj = pd.DataFrame(adj_1) * pd.DataFrame(adj_2)
    # hadamard_adj = np.mat(hadamard_adj)     # 格式转换
    # return hadamard_adj

def pd2spa_tuple(df):
    """
    功能函数：dataframe 转为 稀疏矩阵
    """
    hdm1 = np.mat(df)
    hdm2 = sp.coo_matrix(hdm1)
    hdm3 = sp.csr_matrix(hdm2)
    hdm4 = hdm3.astype(np.float32)
    hdm5 = sparse_to_tuple(hdm4)
    return hdm5

def normalized_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_adj = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    normalized_adj = normalized_adj.astype(np.float32)
    return normalized_adj

def sparse_to_tuple(mx):
    """
    稀疏矩阵转为tuple
    """
    mx = mx.tocoo()
    coords = np.vstack((mx.row, mx.col)).transpose()
    L = tf.SparseTensor(coords, mx.data, mx.shape)
    return tf.sparse_reorder(L)

def calculate_laplacian(adj, lambda_max=1):
    """
    计算拉普拉斯矩阵，输入为带有自连接的矩阵
    """
    # adj = normalized_adj(adj + sp.eye(adj.shape[0]))
    adj = normalized_adj(adj)
    adj = sp.csr_matrix(adj)
    adj = adj.astype(np.float32)
    return sparse_to_tuple(adj)

def weight_variable_glorot(input_dim, output_dim, name=""):
    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    initial = tf.random_uniform([input_dim, output_dim], minval=-init_range,
                            maxval=init_range, dtype=tf.float32)

    return tf.Variable(initial,name=name)

def huanyuan(test1,real1,path,pre_len):
#test处理
    data_test1 = pd.DataFrame()
    a = test1.shape[0]-pre_len
    num = a / 354
    for j in range(432):
        data1 = test1.iloc[:, j]
        ser1 = []
        for i in range(0, 354):
            a = data1[i * num]
            b = data1[i * num + 1]
            mean = (a + b) / 2
            ser1.append(mean)
        data_one1 = pd.DataFrame(ser1)
        data_test1 = pd.concat([data_test1, data_one1], axis=1)
    data_test1.to_csv(path + '/test.csv', encoding='utf-8')
#real处理
    data_real = pd.DataFrame()
    a = data1.shape[0]
    num = a / 354
    for j in range(432):
        data1 = real1.iloc[:, j]
        ser = []
        for i in range(0, 354):
            a = data1[i * num]
            b = data1[i * num + 1]
            mean = (a + b) / 2
            ser.append(mean)
        data_one = pd.DataFrame(ser)
        data_real = pd.concat([data_real, data_one], axis=1)
    data_real.to_csv(path + '/real.csv', encoding='utf-8')
    return data_test1,data_real