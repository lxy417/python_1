# -*- coding: utf-8 -*-
import operator
import pandas as pd
import numpy as np
from tkinter import _flatten

# data_x=pd.read_csv(r'D:\wendang\iris.csv',usecols=[0,1,2,3,4])
# data_y=pd.read_csv(r'D:\wendang\iris.csv',usecols=[5])
# data_x=pd.read_csv(r'D:\wendang\ecoli.csv',usecols=[1,2,3,4,5,6,7],sep='  ')
# data_y=pd.read_csv(r'D:\wendang\ecoli.csv',usecols=[8],sep='  ')
# data_x=pd.read_csv(r'D:\wendang\wine.csv',usecols=[1,2,3,4,5,6,7,8,9,10,11,12,13])
# data_y=pd.read_csv(r'D:\wendang\wine.csv',usecols=[0])
# data_x=pd.read_csv(r'D:\wendang\letter-recognition.csv',usecols=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16])
# data_y=pd.read_csv(r'D:\wendang\letter-recognition.csv',usecols=[0])
data_x = pd.read_csv(r'E:\JetBrains\pycharm\untitled\iris.csv', usecols=[0 ,1, 2, 3])
data_y = pd.read_csv(r'E:\JetBrains\pycharm\untitled\iris.csv', usecols=[4])
train_data = np.array(data_x)  # np.ndarray()
train_x_list = train_data.tolist()  # list
train_data = np.array(data_y)  # np.ndarray()
train_y_list = _flatten(train_data.tolist())  # list


def Knn(A, dataset, labels, k):
    # A为输入样本，dataset为训练集，labels为标签，k为所取范围
    datasetSize = len(dataset)  # datasetSize为样本大小,样本矩阵的行数
    diffMat = np.tile(A, (datasetSize, 1)) - dataset  # 求矩阵差
    sqDiffMat = diffMat ** 2  # 矩阵差平方
    sqDistance = sqDiffMat.sum(axis=1)  # 矩阵差平方的和
    distance = sqDistance ** 0.5  # 测试点与每个样本点的距离
    sortedDistance = distance.argsort()  # 升序排序

    classCount = {}  # 存每个标签的个数
    for i in range(k):
        voteLabel = labels[sortedDistance[i]]  # 得到第i个样本标签
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1  # 对标签的个数计数

    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    # 对标签按个数降序
    return sortedClassCount[0][0]

a=[]
a.append(Knn([6.0,2.5,4.5,1.5], train_x_list, train_y_list, 10))
a.append(Knn([5.0,2.5,7.5,1.5], train_x_list, train_y_list, 10))
a.append(Knn([6.4,2.5,4.5,3.5], train_x_list, train_y_list, 10))
print(a)