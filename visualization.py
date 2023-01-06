#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import pandas as pd
from utils import huanyuan

def plot_result(test_result,test_label1,path,pre_len):
    # 保存数据

    test_result = pd.DataFrame(test_result)
    test_result.to_csv(path + '/test_result.csv',encoding='utf-8')
    test_label1 = pd.DataFrame(test_label1)
    test_label1.to_csv(path + '/test_lable.csv',encoding='utf-8')
    #test_result,test_label1 = huanyuan(test1,real1,path,pre_len)
    ##all test result visualization
    fig1 = plt.figure(figsize=(7,3))
#    ax1 = fig1.add_subplot(1,1,1)
    a_pred = test_result.iloc[:,50]
    a_true = test_label1.iloc[:,50]
    plt.plot(a_pred,'r-',label='prediction')
    plt.plot(a_true,'b-',label='true')
    plt.legend(loc='best',fontsize=10)
    plt.savefig(path+'/test_all.jpg')
    plt.show()
#     ## oneday test result visualization
#     fig1 = plt.figure(figsize=(7,3))
# #    ax1 = fig1.add_subplot(1,1,1)
#     a_pred = test_result.iloc[60:190,50]
#     a_true = test_label1.iloc[60:190,50]
#     plt.plot(a_pred,'r-',label="prediction")
#     plt.plot(a_true,'b-',label="true")
#     plt.legend(loc='best',fontsize=10)
#     plt.savefig(path+'/test_oneday.jpg')
#     plt.show()
    
def plot_error(train_rmse,train_loss,test_rmse,test_acc,test_mae,path):
    ###train_rmse & test_rmse 
    fig1 = plt.figure(figsize=(5,3))
    plt.plot(train_rmse, 'r-', label="train_rmse")
    plt.plot(test_rmse, 'b-', label="test_rmse")
    plt.legend(loc='best',fontsize=10)
    plt.savefig(path+'/rmse.jpg')
    # plt.savefig(path + '/rmse.tif',dpi = 1000)
    # plt.savefig(path + '/rmse.eps',dpi = 1000)
    plt.show()
    #### train_loss & train_rmse
    fig1 = plt.figure(figsize=(5,3))
    plt.plot(train_loss,'b-', label='train_loss')
    plt.legend(loc='best',fontsize=10)
    plt.savefig(path+'/train_loss.jpg')
    # plt.savefig(path + '/train_loss.tif',dpi = 1000)
    # plt.savefig(path + '/train_loss.eps',dpi = 1000)
    plt.show()

    fig1 = plt.figure(figsize=(5,3))
    plt.plot(train_rmse,'b-', label='train_rmse')
    plt.legend(loc='best',fontsize=10)
    plt.savefig(path+'/train_rmse.jpg')
    # plt.savefig(path + '/train_rmse.tif',dpi = 1000)
    # plt.savefig(path + '/train_rmse.eps',dpi = 1000)
    plt.show()

    ### accuracy
    fig1 = plt.figure(figsize=(5,3))
    plt.plot(test_acc, 'b-', label="test_acc")
    plt.legend(loc='best',fontsize=10)
    plt.savefig(path+'/test_acc.jpg')
    # plt.savefig(path + '/test_acc.tif',dpi = 1000)
    # plt.savefig(path + '/test_acc.eps',dpi = 1000)
    plt.show()
    ### rmse
    fig1 = plt.figure(figsize=(5,3))
    plt.plot(test_rmse, 'b-', label="test_rmse")
    plt.legend(loc='best',fontsize=10)
    plt.savefig(path+'/test_rmse.jpg')
    # plt.savefig(path + '/test_rmse.tif',dpi = 1000)
    # plt.savefig(path + '/test_rmse.eps',dpi = 1000)
    plt.show()
    ### mae
    fig1 = plt.figure(figsize=(5,3))
    plt.plot(test_mae, 'b-', label="test_mae")
    plt.legend(loc='best',fontsize=10)
    plt.savefig(path+'/test_mae.jpg')
    # plt.savefig(path + '/test_mae.tif',dpi = 1000)
    # plt.savefig(path + '/test_mae.eps',dpi = 1000)
    plt.show()


