import math
import numpy as np
import pandas as pd

def dtw(file, num1):
    num = num1
    distmatrix = np.zeros((num, num))   #生成一个零矩阵，大小为num*num，方便填数据
    for i in range(num):
        t = file.iloc[:, i]     #取出第i个位置的值
        for j in range(i + 1, num):
            r = file.iloc[:, j]     #去除第i+1位置的值
            if len(r) == 0:
                print("Notdata：", file[j])
            distmatrix[i, j] = mydtw_function2(t, r)    #调用计算函数，计算井i与井j的相似度
            distmatrix[j, i] = distmatrix[i, j]         #(i,j)==(j,i)
            distmatrix[i, i] = 1                        #(i,i)==1
        # print("NO.{0}finish!".format(i))
    return distmatrix

def mydtw_function2(t, r):
    n = len(t)
    m = len(r)
    t = np.array(t)
    r = np.array(r)
    d = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            d[i, j] = np.sum((t[i] - r[j]) ** 2)
    # 累积距离 Cumulative distance
    D = np.ones((n, m)) * np.inf
    D[0, 0] = d[0, 0]
    # 动态规划 dynamic programming
    for i in range(1, n):
        for j in range(m):
            D1 = D[i - 1, j]
            if j > 0:
                D2 = D[i - 1, j - 1]
                D3 = D[i, j - 1]
            else:
                D2 = np.inf
                D3 = np.inf
            D[i, j] = d[i, j] + min([D1, D2, D3])
    dist = D[n - 1, m - 1]
    # 对结果进行处理
    dist = math.exp(-dist)
    return dist


# 描述：使用DTW计算地址相似度的主程序
# 输入：数据集名称（dataname）
def dtw_adj(dataname):
    """
        函数介绍（中文版）
        描述：根据数据集名称读取对应地质特征文件，并在此基础上调用DTW函数计算地址相似性矩阵
        输入：数据集名称(dataname)
        输出：地质相似度矩阵，该矩阵经过了np.mat()转换

        Function introduction (English version)
        Description: Read the corresponding geological feature file according to the dataset name, and call the DTW function to calculate the address similarity matrix
        Input: dataset name (dataname)
        Output: geological similarity matrix, which is transformed by np.mat()
    """

    # 01 读取数据 load_data
    dfall = pd.read_csv(r'data/testdata/'+dataname+'_Details.csv')
    num = dfall.shape[0]

    # 02 归一化 normalization
    dfmean = pd.DataFrame()  # 归一化之后的df
    for i in range(3, dfall.shape[1]):  #取出相关的参数，准备处理
        data1 = dfall.iloc[:, i]
        max_value = np.max(data1)
        min_value = np.min(data1)
        range_value = max_value - min_value
        mean_value = np.mean(data1)
        data1 = (data1 - min_value) / range_value
        dfmean = pd.concat([dfmean, data1], axis=1)  # 参数归一化之后的df

    # 03 计算 calculation
    dfmeant = pd.DataFrame(dfmean.T)    #转置，便于计算
    num1 = dfmeant.shape[1]             #获得行数
    distmatrix = dtw(dfmeant, num1)     #dtw计算主函数
    dfmat = pd.DataFrame(distmatrix)    #输出结果转为DataFrame，便于后续计算

    return dfmat


