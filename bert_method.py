#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 23:36:17 2019

@author: chenjiannan
"""

import numpy as np
import pandas as pd
from keras_bert import load_trained_model_from_checkpoint
from sklearn.model_selection import train_test_split
import pickle
from keras.layers import Input,Dense,SpatialDropout1D,Bidirectional,LSTM,\
                         GlobalAveragePooling1D, Concatenate, GlobalMaxPooling1D,\
                         Lambda, CuDNNLSTM,CuDNNGRU
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Model
from keras import backend as K 
from keras import optimizers
from my_utils import f1

def model_bert_txt(config_path,checkpoint_path,metric = f1, max_txt_len = 100):
    bert_model = load_trained_model_from_checkpoint(config_path, 
                                                    checkpoint_path,
                                                    seq_len=None)
    inp_txt_x1 = Input(shape=(max_txt_len,))
    inp_txt_x2 = Input(shape=(max_txt_len,))
        
    for i in range(20):
        bert_model.layers[-i].trainable
    x1 = bert_model([inp_txt_x1, inp_txt_x2])
    x1 = Lambda(lambda x: x)(x1)
    x1 = SpatialDropout1D(0.3)(x1)
    max_pool = GlobalMaxPooling1D()(x1)
    avg_pool = GlobalAveragePooling1D()(x1)
    pools = Concatenate()([max_pool, avg_pool])
    predictions = Dense(1, activation='sigmoid')(pools)
    model = Model(inputs=[inp_txt_x1, inp_txt_x2],
                  outputs=predictions)
    adam = optimizers.Adam(lr=learning_rate)
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=[metric])
    return model

def model_bert_txt_lstm(config_path,checkpoint_path,metric = f1, max_txt_len = 100):
    bert_model = load_trained_model_from_checkpoint(config_path, 
                                                    checkpoint_path,
                                                    seq_len=None)
    inp_txt_x1 = Input(shape=(max_txt_len,))
    inp_txt_x2 = Input(shape=(max_txt_len,))
        
    for i in range(20):
        bert_model.layers[-i].trainable
    x1 = bert_model([inp_txt_x1, inp_txt_x2])
    x1 = Lambda(lambda x: x)(x1)
    x1 = SpatialDropout1D(0.3)(x1)
    max_pool = GlobalMaxPooling1D()(x1)
    avg_pool = GlobalAveragePooling1D()(x1)
    pools = Concatenate()([max_pool, avg_pool])
    predictions = Dense(1, activation='sigmoid')(pools)
    model = Model(inputs=[inp_txt_x1, inp_txt_x2],
                  outputs=predictions)
    adam = optimizers.Adam(lr=learning_rate)
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=[metric])
    return model

def model_bert_v1(config_path,checkpoint_path,metric = f1):
    bert_model = load_trained_model_from_checkpoint(config_path, 
                                                    checkpoint_path,
                                                    seq_len=None)
    inp_txt_x1 = Input(shape=(None,))
    inp_txt_x2 = Input(shape=(None,))
    inp_entity_x1 = Input(shape=(None,))
    inp_entity_x2 = Input(shape=(None,))
    
    input_entity_txt =  Concatenate()([inp_txt_x1, inp_entity_x1])
    input_entity_txt2 = Concatenate()([inp_txt_x2, inp_entity_x2])
    
    for i in range(10):
        bert_model.layers[-i].trainable
    x_entity = bert_model([inp_entity_x1, inp_entity_x2])
    x_entity = Lambda(lambda x:x)(x_entity)
    x1 = bert_model([input_entity_txt, input_entity_txt2])
    x1 = Lambda(lambda x: x)(x1)
    x1 = SpatialDropout1D(0.3)(x1)
    bilstm = Bidirectional(CuDNNLSTM(256, return_sequences=True, ))
    
    x1 = bilstm(x1)
    x_entity = bilstm(x_entity)

    max_pool1 = GlobalMaxPooling1D()(x1)
    avg_pool1 = GlobalAveragePooling1D()(x1)
    max_pool2 = GlobalMaxPooling1D()(x_entity)
    avg_pool2 = GlobalAveragePooling1D()(x_entity)
    
    pools1 = Concatenate()([max_pool1, avg_pool1,])
    pools2 = Concatenate()([max_pool2, avg_pool2,])
    
    
    pools = Concatenate()([pools1, pools2])
    predictions = Dense(1, activation='sigmoid')(pools)
    model = Model(inputs=[inp_txt_x1, inp_txt_x2,
                          inp_entity_x1, inp_entity_x2],
                  outputs=predictions)
    adam = optimizers.Adam(lr=learning_rate)
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=[metric])
    return model

if __name__ == '__main__':
    config_path = './chinese_L-12_H-768_A-12/bert_config.json'
    checkpoint_path = './chinese_L-12_H-768_A-12/bert_model.ckpt'
    ## 1.1 读取文件
    with open('train_data_model_bert_char.pkl', 'rb') as f:
        train_data = pickle.load(f)
    random_state = 10
    learning_rate = 0.01
    ## 1.2 load data 每次运行
    y = train_data['y_train']
    y_entity = train_data['y_entity_negative']
    X_train_txt = train_data['X_train_txt']
    X_train_title = train_data['X_train_title']
    target_entity_train = train_data['target_entity_train']

    ## 1.3 记录title是否为空
    isnotitle2 =(X_train_title.sum(axis=1)<1)
    ### 只使用title非空的部分
    X_train_title_nonan = X_train_title[~isnotitle2, :]
    y_title_nonan = y[~isnotitle2]
    y_title_nonan_entity = y_entity[~isnotitle2]
    y_index = np.array(range(0,y_title_nonan.shape[0]))
    title_X_train_nonan, title_X_test_nonan, y_train_index, y_test_index = train_test_split(X_train_title_nonan, y_index,
                                                                                            test_size = 0.1,
                                                                                            random_state = random_state)
    ### 获取句子的负面消息 和entity的负面消息
    title_y_train_nonan, title_y_test_nonan = y_title_nonan[y_train_index], y_title_nonan[y_test_index]
    title_y_train_nonan_entity, title_y_test_nonan_entity = y_title_nonan_entity[y_train_index], y_title_nonan_entity[y_test_index]
    
    X_train_txt_cut = X_train_txt[~isnotitle2,:]
    target_entity_train_cut = target_entity_train[~isnotitle2]
    
    txt_X_train_nonan, txt_X_test_nonan = X_train_txt_cut[y_train_index, :], X_train_txt_cut[y_test_index, :]
    target_entity_train_cut_train_nonan, target_entity_train_cut_test_nonan = target_entity_train_cut[y_train_index],target_entity_train_cut[y_test_index]
    
    ## 1.4 have no title
    ### only txt
    y_index = np.array(range(0,X_train_txt.shape[0]))
    X_txt_train, X_txt_test, y_train_index, y_test_index = train_test_split(X_train_txt, y_index,
                                                                            test_size = 0.1,
                                                                            random_state = random_state)
    
    y_train, y_test = y[y_train_index], y[y_test_index]
    y_entity_train, y_entity_test = y_entity[y_train_index], y_entity[y_test_index]
    
    X_target_entity_train, X_target_entity_test = target_entity_train[y_train_index],target_entity_train[y_test_index]
    
#%% 2.load test data, 载入测试集数据，用于预测
    with open('test_data_model_bert_char.pkl', 'rb') as f:
        test_data = pickle.load(f)
    X_test_txt = test_data['X_test_txt']
    X_test_title = test_data['X_test_title']
    target_entity_test = test_data['target_entity_test']

#%% 3.建立bert模型，训练
    config_path = './chinese_L-12_H-768_A-12/bert_config.json'
    checkpoint_path = './chinese_L-12_H-768_A-12/bert_model.ckpt'
    cut_shape = 500
    model = model_bert_txt(config_path,checkpoint_path,metric = f1,
                           max_txt_len = X_txt_train[:,:cut_shape].shape[1])
    
    filepath = "best_weights.h5"
    early_stopping = EarlyStopping(monitor='val_loss',patience=2)
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=0, save_best_only=True, mode='max', period=1)
    callbacks_list = [checkpoint,early_stopping]

    model.fit([X_txt_train[:,:cut_shape], np.zeros(X_txt_train[:,:cut_shape].shape)],
               y_train,
               batch_size = 16,
               epochs = 20,
               validation_data = [[X_txt_test[:,:cut_shape], np.zeros(X_txt_test[:,:cut_shape].shape)],
                                    y_test],
               callbacks=callbacks_list)
    
    
    early_stopping = EarlyStopping(monitor='val_loss',patience=5)
    model_bert_ = model_bert_v1(config_path, checkpoint_path,metric = f1)
    cut_shape = 450
    model_bert_.fit([X_txt_train[:,:cut_shape], np.zeros(X_txt_train[:,:cut_shape].shape),
                    X_target_entity_train[:,:cut_shape], np.zeros(X_target_entity_train[:,:cut_shape].shape)],
                    y_entity_train,
                    batch_size = 32,
                    epochs = 20,
                    validation_data = [[X_txt_test[:,:cut_shape], np.zeros(X_txt_test[:,:cut_shape].shape),
                                        X_target_entity_test[:,:cut_shape], np.zeros(X_target_entity_test[:,:cut_shape].shape)],
                                        y_entity_test],
                    callbacks=[early_stopping])