#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@文件        :trees.py
@说明        :CART算法构造回归树
@时间        :2020/09/22 17:32:59
@作者        :ppka0606
@版本        :1.0
'''


import pandas as pd
import numpy as np


def splitDataset(dataset, feature, threshold):
    """
    将dataset做二分切割，依据的是i >/<= threshold
    i是特征向量中的的第feature项
    """
    smallFeatureValueSet = dataset[dataset[:, feature] <= threshold]
    bigFeatureValueSet = dataset[dataset[:, feature] > threshold]
    return smallFeatureValueSet, bigFeatureValueSet


def calculateError(dataset):
    """
    基于标签（dataset[:,-1])计算误差值
    CART算法基于Gini值计算分散度
    """
    labels = dataset[:, -1]
    labelset = set(labels)
    gini = 1
    for label in labelset:
        gini -= ((labels == label).sum() / np.size(labels)) ** 2
    return gini


def chooseSplitFeatureValue(dataset):
    """
    通过遍历dataset中每个feature上所有的值,进行排序后依次取两个特征值的均值，决定切分最佳的feature(索引)和阈值
    返回划分得到的两个子集，特征索引和阈值
    """

    bestFeatureIndex = 1e10
    bestFeatureThreshold = 1e10
    bestSplitError = 1e10
    bestSubset1 = None
    bestSubset2 = None

    for featureIndex in range(0, np.size(dataset, 1) - 1):
        print(featureIndex)
        featureValueSet = list(set(dataset[:, featureIndex]))
        for valueIndex in range(len(featureValueSet) - 1):
            value = (featureValueSet[valueIndex] +
                     featureValueSet[valueIndex + 1]) / 2
            subset1, subset2 = splitDataset(dataset, featureIndex, value)
            sumError = (np.size(subset1, 0) * calculateError(subset1) +
                        np.size(subset2, 0) * calculateError(subset2)) / np.size(dataset, 0)

            if sumError < bestSplitError:
                bestSplitError = sumError
                bestFeatureIndex = featureIndex
                bestFeatureThreshold = value
                bestSubset1, bestSubset2 = subset1, subset2

    return bestFeatureIndex, bestFeatureThreshold, bestSubset1, bestSubset2


def createTree(dataset):
    """
    通过递归调用的方法构造一个决策树，每一个节点用一个五元的字典表示，
    分别为 分类特征索引、临界值、左子树、右子树、确定的分类标签（若为叶子节点则填写标签，若有多于一类则为-1）、右子树分类

    默认左子树为特征<=阈值，右子树为>阈值
    """
    returnDict = {"featureIndex": None, "threshold": None,
                  "leftTree": None, "rightTree": None, "confirmedLabel": None}
    if dataset is None:
        return None
    elif len(set(dataset[:, -1])) == 1:
        returnDict["confirmedLabel"] = dataset[0][-1]
    else:
        returnDict["confirmedLabel"] = -1
        bestFeatureIndex, bestFeatureThreshold, bestSubset1, bestSubset2 = chooseSplitFeatureValue(
            dataset)
        if len(bestSubset1) != 0 and len(bestSubset2) != 0:
            returnDict["featureIndex"] = bestFeatureIndex
            returnDict["threshold"] = bestFeatureThreshold
            returnDict["leftTree"] = createTree(bestSubset1)
            returnDict["rightTree"] = createTree(bestSubset2)
        else:
            returnDict["confirmedLabel"] = np.bincount(
                (dataset[:, -1]).astype(int)).argmax()
    return returnDict


if __name__ == '__main__':
    df = pd.read_csv('chapter9\Iris-train.txt', sep=' ')
    df_value = df.values

    dataset = np.delete(df_value, -1, 1)
    label = dataset[:, -1]

    tree = createTree(dataset)
    print(tree)

    # self_dataset = np.array([[1, 1, 0],
    #                          [2, 1, 0],
    #                          [1, 2, 1],
    #                          [2, 2, 1]])
    # index, thres, subset1, subset2 = chooseSplitFeatureValue(self_dataset)
    # print(index)
    # print(thres)
    # print(subset1)
    # print(subset2)
    # # print(calculateError(dataset))

    # x = np.array([[1], [2], [3], [2], [1]])
    # print(calculateError(x))
    # feature, threshold = chooseSplitFeatureValue(dataset)
    # a, b = splitDataset(dataset, feature, threshold)

    # print((np.size(a, 0) * calculateError(a) + np.size(b, 0)
    #        * calculateError(b)) / np.size(dataset, 0))
