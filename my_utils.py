#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 10:37:13 2019

@author: chenjiannan
"""
from keras import backend as K 

def f1(y_true, y_pred):
    ''' 由于新版的Keras没有f1可以直接调用，需要自行实现f1的计算，这里使用了backend，
    相当于直接使用tensorflow
    args:
        y_true:真实的标记
        y_pred:预测的结果
    return:
        对应的f1 score
    '''
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def make_deepLearn_data(sentenList, word2idx):
    ''' 将输入的句子tokenize，即将对应的单词标记为对应的下标，
    比如一个句子为['如何', '安全', '提高', '最大化', '收益'],
    标记为[1,2,3,4,5],如果这五个词的对应下标就是这五个数字的化,
    Args:
        sentenList:输入的句子向量，list组成的list，[['如何', '安全', '提高'] ['最大化', '收益']]
        word2idx:词语到对应的标志的字典
    Returns:
        X_train_idx:将句子转化为token后的输入
    '''
    from keras.preprocessing.sequence import pad_sequences
    import numpy as np
    # 确定句子大最大长度
    maxlen = 0
    for i in sentenList:
        if len(i)>maxlen:
            maxlen = len(i)
    
    X_train_idx = [[word2idx.get(w, 0) for w in sen] for sen in sentenList]
    X_train_idx = np.array(pad_sequences(X_train_idx, maxlen, padding='post'))  # 必须是np.array()类型
    return X_train_idx, maxlen