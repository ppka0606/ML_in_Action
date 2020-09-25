#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@文件        :bayes.py
@说明        :基于朴素贝叶斯实现的一个简单文本分类装置
@时间        :2020/09/21 11:05:18
@作者        :ppka0606
@版本        :1.0
'''
# 两个简化的假设：1)特征之间相互独立 2)只使用0/1表示单词是否出现过，而不关注次数
def loadDataSet():
    """
    一个小的测试数据集
    postingList中的每个元素是一句话中的单词切割形成的列表
    classVec表示postingList中每一句话的标签，1-不文明用语；0--无不文明用语
    """
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak',
                       'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]  # 1 is abusive, 0 not
    return postingList, classVec

def createVocabularyList(dataset):
    """
    创建一个集合，收集所有训练集中出现过的单词
    以列表形式返回方便处理
    """
    vocalbularySet = set([])
    for data in dataset:
        vocalbularySet = vocalbularySet | set(data)
    
    return list(vocalbularySet)

