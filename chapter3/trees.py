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
import operator
import pandas as pd

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


def createSelfdataset():
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


'''
@提交说明    :自闭中。。。。。 
@提交时间    :2020/09/15 22:49:43
@提交作者    :ppka0606
'''


def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]  # chop out axis used for splitting
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet


def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1  # the last column is used for the labels
    baseEntropy = calculateShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):  # iterate over all the features
        # create a list of all the examples of this feature
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)  # get a set of unique values
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob * calculateShannonEnt(subDataSet)
        # calculate the info gain; ie reduction in entropy
        infoGain = baseEntropy - newEntropy
        if (infoGain > bestInfoGain):  # compare this to the best gain so far
            bestInfoGain = infoGain  # if better than current best, set to best
            bestFeature = i

    return bestFeature  # returns an integer


def majorityCnt(classList):
    """
    返回由所有标签构成的列表中，出现次数最多的标签
    """
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(),
                              key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def createTree(dataSet, labels):
	"""
	Function：	创建树
	Args：		dataSet：数据集
				labels：标签列表
	Returns：	myTree：创建的树的信息
	"""
	#创建分类列表
	classList = [example[-1] for example in dataSet]
	#类别完全相同则停止划分
	if classList.count(classList[0]) == len(classList):
		return classList[0]
	#遍历完所有特征时返回出现次数最多的类别
	if len(dataSet[0]) == 1:
		return majorityCnt(classList)
	#选取最好的分类特征
	bestFeat = chooseBestFeatureToSplit(dataSet)
	bestFeatLabel = labels[bestFeat]
	#创建字典存储树的信息
	myTree = {bestFeatLabel: {}}
	del(labels[bestFeat])
	#得到列表包含的所有属性值
	featValues = [example[bestFeat] for example in dataSet]
	#从列表中创建集合
	uniqueVals = set(featValues)
	#遍历当前选择特征包含的所有属性值
	for value in uniqueVals:
		#复制类标签
		subLabels = labels[:]
		#递归调用函数createTree()，返回值将被插入到字典变量myTree中
		myTree[bestFeatLabel][value] = createTree(
		    splitDataSet(dataSet, bestFeat, value), subLabels)
	#返回字典变量myTree
	return myTree


if __name__ == '__main__':
    data = pd.read_table('chapter3\Iris-train.txt', sep = '\n', header = None)
    labels = ['feature1', 'feature2', 'feature3', 'feature4']
    print(data)

    
'''
@修改说明    :现在可以针对简单的标签进行分类，但是对于iris数据集中的连续型变量，必须更换算法
@修改时间    :2020/09/21 10:04:26
@修改作者    :ppka0606
'''
    
    