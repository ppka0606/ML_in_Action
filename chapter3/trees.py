#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@文件        :trees.py
@说明        :
@时间        :2020/09/15 21:57:06
@作者        :ppka0606
@版本        :1.0
'''


from math import log


def calculateShannonEnt(dataset):
    """
    计算给定数据集的香农熵
    """
    totalnum = len(dataset)
    labelDict = {}

    print(type(dataset))

    # 统计不同标签下的数据组数
    for data in dataset:
        label = data[-1]
        if label in labelDict.keys():
            labelDict[label] += 1
        else:
            labelDict[label] = 1

    result = 0
    for item in labelDict.values():
        result += float((-1) * item / totalnum * log(item / totalnum, 2))

    return result


def createSelfDataset():
    """
    自行创建一个数据集用于测试
    """
    dataset = [[1, 1, 1],
               [1, 1, 1],
               [1, 0, 0],
               [0, 1, 0],
               [0, 1, 0]]
    labels = ["no surfacing", "flippers"]
    return dataset, labels

dat,lab = createSelfDataset()
calculateShannonEnt(dat)

'''
@提交说明    :自闭中。。。。。 
@提交时间    :2020/09/15 22:49:43
@提交作者    :ppka0606
'''


