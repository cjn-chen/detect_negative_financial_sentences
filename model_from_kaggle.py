# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 19:03:37 2019

@author: cjn
"""

from keras.layers import Input, Embedding, SpatialDropout1D, Bidirectional, \
                         GlobalMaxPooling1D, CuDNNLSTM, CuDNNGRU, Concatenate,\
                         Dense
from keras.models import Model
from keras import optimizers
from keras import backend as K  #调用后端引擎，K相当于使用tensorflow（后端是tf的话）
import pandas as pd
from word_model2embeding_matrix import make_deepLearn_data, split_word
import pickle
from sklearn.model_selection import train_test_split

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

def build_model(embedding_matrix, learning_rate, nb_words,
                max_length=55,embedding_size=300, metric = f1):
    '''
    根据预训练的嵌入矩阵，返回神经网络的模型，返回模型还需要调用model.fit模块
    Args:
        embedding_matrix:嵌入矩阵,每行为一个单词，每列为其中一个维度
        nb_words:词汇表大小，设置为出现过的词汇数目+1，空的位置留给OOV(out of vocabulary),
        max_length:
    '''
    inp = Input(shape=(max_length,))  # 定义输入
    # 嵌入层
    x = Embedding(nb_words, embedding_size, weights=[embedding_matrix], trainable=False)(inp)
    x = SpatialDropout1D(0.3)(x)  #　对某一个维度进行dropout,embedding中的某一列
    x1 = Bidirectional(CuDNNLSTM(256, return_sequences=True))(x)
    x2 = Bidirectional(CuDNNGRU(128, return_sequences=True))(x1)
    max_pool1 = GlobalMaxPooling1D()(x1)
    max_pool2 = GlobalMaxPooling1D()(x2)
    conc = Concatenate()([max_pool1, max_pool2])
    predictions = Dense(1, activation='sigmoid')(conc)
    model = Model(inputs=inp, outputs=predictions)
    adam = optimizers.Adam(lr=learning_rate)
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=[f1])
    return model


if __name__ == '__main__':
    print('Loading data...')
    ## loda data from file
    with open('word2idx_embedMatrix.pkl', 'rb') as f:
        word2idx, embedMatrix = pickle.load(f)
    with open('train_data.pkl', 'rb') as f:
        train_data = pickle.load(f)
    ## load data 
    y = train_data['y_train']
    X_train_txt, X_train_txt_max_len = train_data['X_train_txt'],train_data['X_train_txt_max_len']
    X_train_title, X_train_title_max_len = train_data['X_train_title'],train_data['X_train_title_max_len']
    
    ## split data txt
    X_train, X_test, y_train, y_test = train_test_split(X_train_txt, y,
                                                        test_size = 0.1,
                                                        random_state = 0)
    nb_words = len(word2idx.keys()) + 1
    learning_rate = 0.01
    max_len = X_train.shape[1]
    model = build_model(embedMatrix, learning_rate, nb_words,
                        max_length = max_len,
                        embedding_size = embedMatrix.shape[1])
    
    model.fit(X_train, y_train,
              batch_size=32,
              epochs=5,
              validation_data=[X_test, y_test])
    
    ## split data title
    X_train, X_test, y_train, y_test = train_test_split(X_train_title, y,
                                                        test_size = 0.1,
                                                        random_state = 0)
    max_len = X_train.shape[1]
    model = build_model(embedMatrix, learning_rate, nb_words,
                        max_length = max_len,
                        embedding_size = embedMatrix.shape[1])
    model.fit(X_train, y_train,
              batch_size=32,
              epochs=20,
              validation_data=[X_test, y_test])
