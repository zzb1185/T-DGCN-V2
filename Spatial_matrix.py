import math
import numpy as np
import pandas as pd

def spa_adj(dataname):
    dfall = pd.read_csv(r'data/testdata/'+dataname+'_Details.csv')
    dfxy = dfall.iloc[:, 1:3]  # 取出经纬度计算
    # 01 计算邻近关系
    num = dfxy.shape[0]  # 多少口井
    dfnear = pd.DataFrame(columns=['a', 'b'])  # 临近矩阵
    for i in range(num):
        x1 = dfxy.iloc[i, 0]
        y1 = dfxy.iloc[i, 1]
        for j in range(num):
            x2 = dfxy.iloc[j, 0]
            y2 = dfxy.iloc[j, 1]
            dis = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** (0.5)
            if dis <= 350:  #350m
                dfnear = dfnear.append(pd.DataFrame({'a': [i], 'b': [j]}))  # 临近矩阵
    print("邻近关系计算完成")

    # 02 组合
    num_finished = 1
    oh432 = np.zeros((num, num))
    for row in range(dfnear.shape[0]):
        input = dfnear.iloc[(row, 0)]
        near = dfnear.iloc[(row, 1)]
        oh432[input][near] = 1
        print(row)

    dfoh1 = pd.DataFrame(oh432)  # 完全体权重邻接矩阵
    dfoh1.to_csv(r"data\testdata\test_Spamatrix.csv",index=False,header=None)
    print("矩阵组装完成，保存在data中")
    return dfoh1
