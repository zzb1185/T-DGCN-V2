# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 15:15:50 2018

@author: Administrator
"""

import numpy as np
import pandas as pd
import pickle as pkl

def load_testtdgcn_data(dataset):
    cq504_adj = pd.read_csv(r'data/testdata/test_OHe.csv', header=None)
    adj = np.mat(cq504_adj)
    cq504_cql = pd.read_csv(r'data/testdata/CBM_Production.csv')
    cq504_cql = cq504_cql.fillna(0)
    return cq504_cql, adj

def load_testtgcn_data(dataset):
    cq504_adj = pd.read_csv(r'data/testdata/test_OH1.csv', header=None)
    adj = np.mat(cq504_adj)
    cq504_cql = pd.read_csv(r'data/testdata/CBM_Production.csv')
    cq504_cql = cq504_cql.fillna(0)
    return cq504_cql, adj

def load_test_data(dataset):
    spametrics = pd.read_csv(r'data/testdata/test_Spamatrix.csv', header=None)
    spametrics = np.mat(spametrics)
    prod = pd.read_csv(r'data/testdata/CBM_Production.csv')
    prod = prod.fillna(0)
    return prod, spametrics


def preprocess_data(data, time_len, rate, seq_len, pre_len):
    train_size = int(time_len * rate)
    train_data = data[0:train_size]
    test_data = data[train_size:time_len]
    trainX, trainY, testX, testY = [], [], [], []
    for i in range(len(train_data) - seq_len - pre_len):
        a = train_data[i: i + seq_len + pre_len]
        trainX.append(a[0 : seq_len])
        trainY.append(a[seq_len : seq_len + pre_len])
    for i in range(len(test_data) - seq_len -pre_len):
        b = test_data[i: i + seq_len + pre_len]
        testX.append(b[0 : seq_len])
        testY.append(b[seq_len : seq_len + pre_len])
      
    trainX1 = np.array(trainX)
    trainY1 = np.array(trainY)
    testX1 = np.array(testX)
    testY1 = np.array(testY)
    return trainX1, trainY1, testX1, testY1
    
