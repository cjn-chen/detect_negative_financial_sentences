# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 19:03:37 2019

@author: cjn
"""

from keras.layers import Input, Embedding, SpatialDropout1D, Bidirectional, \
                         GlobalMaxPooling1D, CuDNNLSTM, CuDNNGRU, Concatenate,\
                         Dense, GlobalAveragePooling1D
from keras.models import Model
from keras import optimizers
from keras import backend as K  #调用后端引擎，K相当于使用tensorflow（后端是tf的话）
import pickle
import numpy as np
from keras.preprocessing.sequence import pad_sequences

def get_target_entity_position(target_entity_list, txt_X, title_X_train):
    entity_txt_title = np.hstack((txt_X, title_X_train))
    position_relate = []
    i = 0
    for entity in target_entity_list:
        entity_row = entity_txt_title[i,:]
        tmp_args = np.argwhere(entity_row==entity)
        if tmp_args.shape[0] >  0:
            position_relate.append(tmp_args[0][0])
        else:
            position_relate.append(0)
    return np.array(position_relate)

def get_target_entity_position2(target_entity_list, txt_X,):
    position_relate = []
    i = 0
    for entity in target_entity_list:
        entity_row = txt_X[i,:]
        tmp_args = np.argwhere(entity_row==entity)
        if tmp_args.shape[0] >  0:
            position_relate.append(tmp_args[0][0])
        else:
            position_relate.append(0)
    return np.array(position_relate)

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
    x = Embedding(nb_words, embedding_size, weights=[embedding_matrix], trainable=True)(inp)
    x = SpatialDropout1D(0.3)(x)  # 对某一个维度进行dropout,embedding中的某一列
    x1 = Bidirectional(CuDNNLSTM(256, return_sequences=True))(x)  # 使用GPU加速的LSTM
    x2 = Bidirectional(CuDNNGRU(128, return_sequences=True))(x1)  # 使用GPU加速的GRU
    max_pool1 = GlobalMaxPooling1D()(x1)  #对于时序数据的全局最大池化，
    max_pool2 = GlobalMaxPooling1D()(x2)  #对于时序数据的全局最大池化。
    avg_pool1 = GlobalAveragePooling1D()(x1)
    avg_pool2 = GlobalAveragePooling1D()(x2)
    conc = Concatenate()([max_pool1, max_pool2,avg_pool1,avg_pool2])  # 合并两层
    predictions = Dense(1, activation='sigmoid')(conc)
    model = Model(inputs=inp, outputs=predictions)
    adam = optimizers.Adam(lr=learning_rate)
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=[metric])
    return model

def build_model_title_txt(embedding_matrix, learning_rate, nb_words,
                          max_length=55,max_length_txt = 100,
                          embedding_size=300, metric = f1):
    '''
    根据预训练的嵌入矩阵，返回神经网络的模型，返回模型还需要调用model.fit模块
    Args:
        embedding_matrix:嵌入矩阵,每行为一个单词，每列为其中一个维度
        nb_words:词汇表大小，设置为出现过的词汇数目+1，空的位置留给OOV(out of vocabulary),
        max_length:
    '''
    inp = Input(shape=(max_length,))  # 定义输入
    inp_txt = Input(shape=(max_length_txt,))  # 定义输入
    # 嵌入层
    embed = Embedding(nb_words, embedding_size, weights=[embedding_matrix], trainable=True)
    # title
    x = embed(inp)
    x = SpatialDropout1D(0.3)(x)  # 对某一个维度进行dropout,embedding中的某一列
    x1 = Bidirectional(CuDNNLSTM(128, return_sequences=True))(x)  # 使用GPU加速的LSTM
    x2 = Bidirectional(CuDNNGRU(64, return_sequences=True))(x1)  # 使用GPU加速的GRU
    max_pool1 = GlobalMaxPooling1D()(x1)  #对于时序数据的全局最大池化，
    max_pool2 = GlobalMaxPooling1D()(x2)  #对于时序数据的全局最大池化。
    avg_pool1 = GlobalAveragePooling1D()(x1)
    avg_pool2 = GlobalAveragePooling1D()(x2)
    # txt
    x_txt = embed(inp_txt)
    x_txt = SpatialDropout1D(0.3)(x_txt)  # 对某一个维度进行dropout,embedding中的某一列
    x1_txt = Bidirectional(CuDNNLSTM(128, return_sequences=True))(x_txt)  # 使用GPU加速的LSTM
    x2_txt = Bidirectional(CuDNNGRU(64, return_sequences=True))(x1_txt)  # 使用GPU加速的GRU
    max_pool1_txt = GlobalMaxPooling1D()(x1_txt)  #对于时序数据的全局最大池化，
    max_pool2_txt = GlobalMaxPooling1D()(x2_txt)  #对于时序数据的全局最大池化。
    avg_pool1_txt = GlobalAveragePooling1D()(x1_txt)
    avg_pool2_txt = GlobalAveragePooling1D()(x2_txt)
    
    conc = Concatenate()([max_pool1, max_pool2,avg_pool1,avg_pool2,
                          max_pool1_txt, max_pool2_txt, avg_pool1_txt, avg_pool2_txt,])  # 合并两层
    
    predictions = Dense(1, activation='sigmoid')(conc)

    model = Model(inputs=[inp,inp_txt], outputs=predictions)
    adam = optimizers.Adam(lr=learning_rate)
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=[metric])
    return model

#def build_model_stack_title_predict(embedding_matrix, learning_rate, nb_words,
#                                    max_length=55,embedding_size=300, metric = f1):
#    '''
#    根据预训练的嵌入矩阵，返回神经网络的模型，返回模型还需要调用model.fit模块
#    Args:
#        embedding_matrix:嵌入矩阵,每行为一个单词，每列为其中一个维度
#        nb_words:词汇表大小，设置为出现过的词汇数目+1，空的位置留给OOV(out of vocabulary),
#        max_length:
#    '''
#    inp = Input(shape=(max_length,))  # 定义输入
#    inp_title_predict = Input(shape=(1,))  # title的预测结果
#    # 嵌入层
#    x = Embedding(nb_words, embedding_size, weights=[embedding_matrix], trainable=True)(inp)
#    x = SpatialDropout1D(0.3)(x)  # 对某一个维度进行dropout,embedding中的某一列
#    x1 = Bidirectional(CuDNNLSTM(256, return_sequences=True))(x)  # 使用GPU加速的LSTM
#    x2 = Bidirectional(CuDNNGRU(128, return_sequences=True))(x1)  # 使用GPU加速的GRU
#    max_pool1 = GlobalMaxPooling1D()(x1)  #对于时序数据的全局最大池化，
#    max_pool2 = GlobalMaxPooling1D()(x2)  #对于时序数据的全局最大池化
#    conc = Concatenate()([max_pool1, max_pool2, inp_title_predict])  # 合并两层
#    
#    predictions = Dense(1, activation='sigmoid')(conc)
#    
##    conc_pred = Concatenate()([predictions, inp_title_predict])  # 合并两个预测结果
##    predictions_end = Dense(1, activation='sigmoid')(conc_pred)
##    weight_1 = Lambda(lambda x:x*0.5)
##    weight_2 = Lambda(lambda x:x*0.5)
##    weight_pred1 = weight_1(predictions)
##    weight_pred2 = weight_2(inp_title_predict)
##    last = Add()([weight_pred1,weight_pred2])
#    
#    model = Model(inputs = [inp, inp_title_predict], outputs=predictions)
#    adam = optimizers.Adam(lr=learning_rate)
#    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=[metric])
#    return model

#def build_model_add_title2(embedding_matrix, learning_rate, nb_words,
#                          max_length=55, max_length_title=55,
#                          embedding_size=200, metric = f1):
#    '''
#    根据预训练的嵌入矩阵，返回神经网络的模型，返回模型还需要调用model.fit模块
#    Args:
#        embedding_matrix:嵌入矩阵,每行为一个单词，每列为其中一个维度
#        learning_rate:学习率的大小
#        nb_words:词汇表大小，设置为出现过的词汇数目+1，空的位置留给OOV(out of vocabulary),
#        max_length:txt中句子的最大长度
#        max_length_title:title中句子的最大长度
#        embedding_size:嵌入矩阵的嵌入维度，即嵌入矩阵embedding_matrix.shape[1]
#        metric:使用的评价方式
#    '''
#    inp = Input(shape=(max_length,))  # 定义输入 txt
#    inp_title = Input(shape=(max_length_title,))  # 定义输入 title
#
#    # txt
#    x = Embedding(nb_words, embedding_size, weights=[embedding_matrix], trainable=True)(inp)# 嵌入层
#    x = SpatialDropout1D(0.3)(x)  # 对某一个维度进行dropout,embedding中的某一列
#    x1 = Bidirectional(CuDNNLSTM(256, return_sequences=True))(x)  # 使用GPU加速的LSTM
#    x2 = Bidirectional(CuDNNGRU(128, return_sequences=True))(x1)  # 使用GPU加速的GRU
#    max_pool1 = GlobalMaxPooling1D()(x1)  #对于时序数据的全局最大池化
#    max_pool2 = GlobalMaxPooling1D()(x2)  #对于时序数据的全局最大池化
#    conc = Concatenate()([max_pool1, max_pool2])  # 合并两层
##    conc = Dropout(0.3)(conc)
#    # title
#    x_title = Embedding(nb_words, embedding_size, weights=[embedding_matrix], trainable=False)(inp_title)# 嵌入层
#    x_title = SpatialDropout1D(0.3)(x_title)  # 对某一个维度进行dropout,embedding中的某一列
#    x1_title = Bidirectional(CuDNNLSTM(256, return_sequences=True))(x_title)  # 使用GPU加速的LSTM
#    x2_title = Bidirectional(CuDNNGRU(128, return_sequences=True))(x1_title)  # 使用GPU加速的GRU
#    max_pool1_title = GlobalMaxPooling1D()(x1_title)  #对于时序数据的全局最大池化，
#    max_pool2_title = GlobalMaxPooling1D()(x2_title)  #对于时序数据的全局最大池化。
#    conc_title = Concatenate()([max_pool1_title, max_pool2_title])  # 合并两层
##    conc_title = Dropout(0.3)(conc_title)
#    
#    conc_all = Concatenate()([conc, conc_title])
#    
#    predictions = Dense(1, activation='sigmoid')(conc_all)
#    model = Model(inputs=[inp, inp_title], outputs=predictions)
#    adam = optimizers.Adam(lr=learning_rate)
#    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=[metric])
#    return model


#def build_model_add_title_merge(embedding_matrix, learning_rate, nb_words,
#                                max_length=55, max_length_title=55,
#                                embedding_size=200, metric = f1):
#    '''
#    根据预训练的嵌入矩阵，返回神经网络的模型，返回模型还需要调用model.fit模块
#    Args:
#        embedding_matrix:嵌入矩阵,每行为一个单词，每列为其中一个维度
#        learning_rate:学习率的大小
#        nb_words:词汇表大小，设置为出现过的词汇数目+1，空的位置留给OOV(out of vocabulary),
#        max_length:txt中句子的最大长度
#        max_length_title:title中句子的最大长度
#        embedding_size:嵌入矩阵的嵌入维度，即嵌入矩阵embedding_matrix.shape[1]
#        metric:使用的评价方式
#    '''
#    inp = Input(shape=(max_length,))  # 定义输入 txt
#    inp_title = Input(shape=(max_length_title,))  # 定义输入 title
#    inp_entity = Input(shape=(1,)) # 需要检测的entity
#    # 合并三个输入
#    concate_input = Concatenate()([inp_entity, inp_title, inp])
#    # 嵌入层
#    embed = Embedding(nb_words, embedding_size, weights=[embedding_matrix], trainable=True)
#    # 进行embedding
#    embedding = embed(concate_input)
#    # 进行attention操作
#    x = Attention(8,16)([embedding, embedding, embedding])
#    # txt
#    x = SpatialDropout1D(0.3)(x)  # 对某一个维度进行dropout,embedding中的某一列
#    x1 = Bidirectional(CuDNNLSTM(256, return_sequences=True))(x)  # 使用GPU加速的LSTM
#    x2 = Bidirectional(CuDNNGRU(128, return_sequences=True))(x1)  # 使用GPU加速的GRU
#    max_pool1 = GlobalMaxPooling1D()(x1)  #对于时序数据的全局最大池化
#    max_pool2 = GlobalMaxPooling1D()(x2)  #对于时序数据的全局最大池化
#    conc = Concatenate()([max_pool1, max_pool2])  # 合并两层
##    conc = Dropout(0.3)(conc)
#    predictions = Dense(1, activation='sigmoid')(conc)
#    model = Model(inputs=[inp, inp_title, inp_entity], outputs=predictions)
#    adam = optimizers.Adam(lr=learning_rate)
#    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=[metric])
#    return model


#def build_model_add_title_merge2(embedding_matrix, learning_rate, nb_words,
#                                max_length=55, max_length_title=55,
#                                embedding_size=200, metric = f1):
#    '''
#    根据预训练的嵌入矩阵，返回神经网络的模型，返回模型还需要调用model.fit模块
#    Args:
#        embedding_matrix:嵌入矩阵,每行为一个单词，每列为其中一个维度
#        learning_rate:学习率的大小
#        nb_words:词汇表大小，设置为出现过的词汇数目+1，空的位置留给OOV(out of vocabulary),
#        max_length:txt中句子的最大长度
#        max_length_title:title中句子的最大长度
#        embedding_size:嵌入矩阵的嵌入维度，即嵌入矩阵embedding_matrix.shape[1]
#        metric:使用的评价方式
#    '''
#    inp = Input(shape=(max_length,))  # 定义输入 txt
#    inp_title = Input(shape=(max_length_title,))  # 定义输入 title
#    inp_entity = Input(shape=(1,)) # 需要检测的entity
#    inp_title_predict = Input(shape=(1,)) # 需要检测的entity
#    inp_text_predict = Input(shape=(1,)) # 需要检测的entity
#    inp_entity_position = Input(shape=(1,))
#    # 合并三个输入
#    concate_input = Concatenate()([inp, inp_title, inp_entity])
#    # 嵌入层
#    embed = Embedding(nb_words, embedding_size, weights=[embedding_matrix], trainable=True)
#    # 进行embedding
#    embedding = embed(concate_input)
#    embed_pos = TrainablePositionEmbedding(nb_words,embedding_size)
#    
#    embedding = embed_pos([embedding,inp_entity_position])
#    # txt
#    x = SpatialDropout1D(0.3)(embedding)  # 对某一个维度进行dropout,embedding中的某一列
#    
#    bilstm = Bidirectional(CuDNNLSTM(64, return_sequences=True))  # 使用GPU加速的LSTM
#    x1 = bilstm(x)
###    
#    x2 = Bidirectional(CuDNNGRU(32, return_sequences=True))(x)  # 使用GPU加速的GRU
#    max_pool1 = GlobalMaxPooling1D()(x1)  #对于时序数据的全局最大池化
#    max_pool2 = GlobalMaxPooling1D()(x2)  #对于时序数据的全局最大池化
#    avg_pool1 = GlobalAveragePooling1D()(x1)
#    avg_pool2 = GlobalAveragePooling1D()(x2)
#    
#    concate_pool = Concatenate()([max_pool1, max_pool2,
#                                  avg_pool1, avg_pool2])
#    
#    predictions1 = Dense(30, activation='sigmoid')(concate_pool)
#    
#    conc = Concatenate()([inp_title_predict,inp_text_predict,
#                          predictions1])  # 合并两层
#
#    predictions = Dense(1, activation='sigmoid')(conc)
#    model = Model(inputs=[inp, inp_title, inp_entity,
#                          inp_title_predict,inp_text_predict,
#                          inp_entity_position], outputs=predictions)
#    adam = optimizers.Adam(lr=learning_rate)
#    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=[metric])
#    return model
#
#def build_model_add_title_merge3(embedding_matrix, learning_rate, nb_words,
#                                max_length=55, max_length_title=55,
#                                embedding_size=200, metric = f1):
#    '''
#    根据预训练的嵌入矩阵，返回神经网络的模型，返回模型还需要调用model.fit模块
#    Args:
#        embedding_matrix:嵌入矩阵,每行为一个单词，每列为其中一个维度
#        learning_rate:学习率的大小
#        nb_words:词汇表大小，设置为出现过的词汇数目+1，空的位置留给OOV(out of vocabulary),
#        max_length:txt中句子的最大长度
#        max_length_title:title中句子的最大长度
#        embedding_size:嵌入矩阵的嵌入维度，即嵌入矩阵embedding_matrix.shape[1]
#        metric:使用的评价方式
#    '''
#    inp = Input(shape=(max_length,))  # 定义输入 txt
#    inp_entity = Input(shape=(1,)) # 需要检测的entity
#    inp_text_predict = Input(shape=(1,)) # 需要检测的entity
#    inp_entity_position = Input(shape=(1,))
#    # 合并三个输入
#    concate_input = Concatenate()([inp, inp_entity])
#    # 嵌入层
#    embed = Embedding(nb_words, embedding_size, weights=[embedding_matrix], trainable=True)
#    # 进行embedding
#    embedding = embed(concate_input)
#    embed_pos = TrainablePositionEmbedding(nb_words,embedding_size)
#    
#    embedding = embed_pos([embedding,inp_entity_position])
#    # txt
#    x = SpatialDropout1D(0.3)(embedding)  # 对某一个维度进行dropout,embedding中的某一列
#    
#    bilstm = Bidirectional(CuDNNLSTM(64, return_sequences=True))  # 使用GPU加速的LSTM
#    x1 = bilstm(x) 
#    x2 = Bidirectional(CuDNNGRU(64, return_sequences=True))(x)  # 使用GPU加速的GRU
#    
#    max_pool1 = GlobalMaxPooling1D()(x1)  #对于时序数据的全局最大池化
#    max_pool2 = GlobalMaxPooling1D()(x2)  #对于时序数据的全局最大池化
#    avg_pool1 = GlobalAveragePooling1D()(x1)
#    avg_pool2 = GlobalAveragePooling1D()(x2)
#    
#    concate_pool = Concatenate()([max_pool1, max_pool2,
#                                  avg_pool1, avg_pool2])
#    
#    predictions1 = Dense(20, activation='sigmoid')(concate_pool)
#    
#    conc = Concatenate()([inp_text_predict,
#                          predictions1])  # 合并两层
#
#    predictions = Dense(1, activation='sigmoid')(conc)
#    model = Model(inputs=[inp, inp_entity,
#                          inp_text_predict,
#                          inp_entity_position], outputs=predictions)
#    adam = optimizers.Adam(lr=learning_rate)
#    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=[metric])
#    return model

if __name__ == '__main__':
    print('Loading data...')
#%% 1. loda train data from file
    ## 1.1 读取文件
    ### 读取嵌入矩阵---------------------------------------
    with open('word2idx_embedMatrix.pkl', 'rb') as f:
        word2idx_notitle, embedMatrix_notitle = pickle.load(f)
    with open('word2idx_embedMatrix_hastitle.pkl', 'rb') as f:
        word2idx_hastitle, embedMatrix_hastitle = pickle.load(f)   
    # 有title的部分的字典中，词的数目
    nb_words_hastitle = len(word2idx_hastitle.keys()) + 1
    # 没有title的部分的字典中，词的数目
    nb_words_notitle = len(word2idx_notitle.keys()) + 1
    ### 读取训练集和测试集---------------------------------
    with open('train_data_model.pkl', 'rb') as f:
        train_data_notitle = pickle.load(f)
    with open('train_data_model_hastitle.pkl', 'rb') as f:
        train_data_hastitle = pickle.load(f)

#%% 2.load test data, 载入测试集数据，用于预测
    with open('test_data_model.pkl', 'rb') as f:
        test_data_notitle = pickle.load(f)
    with open('test_data_model_hastitle.pkl', 'rb') as f:
        test_data_hastitle = pickle.load(f)
#%% 3.train the model and predict data
#%% 3.1模型的超参数
    random_state = 10
    learning_rate = 0.01
    # 选择95%的txt的长度进行截断
    all_txt_len = np.append((train_data_hastitle['txt']>0).sum(axis=1),(train_data_notitle['txt']>0).sum(axis=1))
    cut_shape_txt = int(np.percentile(all_txt_len,95))

    if train_data_hastitle['txt'].shape[1] > cut_shape_txt:
        train_data_hastitle['txt'] = train_data_hastitle['txt'][:,:cut_shape_txt]
        if test_data_hastitle['txt'].shape[1] >= cut_shape_txt:
            test_data_hastitle['txt'] = test_data_hastitle['txt'][:,:cut_shape_txt]
        else:
            test_data_hastitle['txt'] = pad_sequences(test_data_hastitle['txt'],
                                                      cut_shape_txt, padding='post')
    # 选择95%的txt的长度进行截断
    if train_data_notitle['txt'].shape[1] > cut_shape_txt:
        train_data_notitle['txt'] = train_data_notitle['txt'][:,:cut_shape_txt]
        if test_data_notitle['txt'].shape[1] >= cut_shape_txt:
            test_data_notitle['txt'] = test_data_notitle['txt'][:,:cut_shape_txt]
        else:
            test_data_notitle['txt'] = pad_sequences(test_data_notitle['txt'],
                                                     cut_shape_txt, padding='post')    

#%% 3.2 (处理含有title的)title_predict和txt_predict，和最终模型混合==============
    print('built model ...')
    ## title--------------------------------------------------------------
#    early_stopping = EarlyStopping(monitor='val_loss',patience=1)
    model_title = build_model_title_txt(embedMatrix_hastitle, learning_rate, nb_words_hastitle,
                                        max_length = train_data_hastitle['title'].shape[1],
                                        max_length_txt = train_data_hastitle['txt'].shape[1],
                                        embedding_size = embedMatrix_hastitle.shape[1])
    model_title.fit([train_data_hastitle['title'],train_data_hastitle['txt']],
                    train_data_hastitle['y_train'],
                    batch_size = 32,
                    epochs = 2,
#                    callbacks = [early_stopping],
#                    validation_split = 0.2,
                    )
    
    title_predict_hastitle = model_title.predict([train_data_hastitle['title'],train_data_hastitle['txt']])
    ### predict test set
    title_predict_hastitle_test = model_title.predict([test_data_hastitle['title'],test_data_hastitle['txt']])
    K.clear_session()
    ## txt-----------------------------------------------------------------
#    model_txt = build_model(embedMatrix_hastitle, learning_rate, nb_words_hastitle,
#                            max_length = train_data_hastitle['txt'].shape[1],
#                            embedding_size = embedMatrix_hastitle.shape[1])
#    model_txt.fit(train_data_hastitle['txt'], train_data_hastitle['y_train'],
#                  batch_size = 32,
#                  epochs = 2)
    
    model_txt_notitle = build_model(embedMatrix_notitle, learning_rate, nb_words_notitle,
                                    max_length = train_data_notitle['txt'].shape[1],
                                    embedding_size = embedMatrix_notitle.shape[1])
    model_txt_notitle.fit(train_data_notitle['txt'], train_data_notitle['y_train'],
                          batch_size = 32,
                          epochs = 2,)
    txt_predict_notitle = model_txt_notitle.predict(train_data_notitle['txt'])
    ### predict test set
    txt_predict_notitle_test = model_txt_notitle.predict(test_data_notitle['txt'])
#    
#    txt_predict_hastitle = model_txt_notitle.predict(train_data_hastitle['txt'])
##    ### predict test set
#    txt_predict_hastitle_test = model_txt_notitle.predict(test_data_hastitle['txt'])
    K.clear_session()
    ## merge model
#    model_merge = build_model_add_title_merge2(embedMatrix_hastitle, learning_rate, 
#                                               nb_words_hastitle,
#                                               max_length = train_data_hastitle['txt'].shape[1],
#                                               max_length_title = train_data_hastitle['title'].shape[1],
#                                               embedding_size = embedMatrix_hastitle.shape[1],
#                                               metric = f1)
#    
#    entity_position_train = get_target_entity_position(train_data_hastitle['target_entity'],
#                                                       train_data_hastitle['txt'],
#                                                       train_data_hastitle['title'])
#
#    model_merge.fit([train_data_hastitle['txt'], train_data_hastitle['title'],
#                     train_data_hastitle['target_entity'],
#                     title_predict_hastitle, txt_predict_hastitle,
#                     entity_position_train], 
#                    train_data_hastitle['y_entity_negative'],
#                    batch_size=32,
#                    epochs= 5)
#    
#    ## 预测测试集
#    entity_position_test = get_target_entity_position(test_data_hastitle['target_entity'],
#                                                            test_data_hastitle['txt'],
#                                                            test_data_hastitle['title'])
#    
#    predict_hastitle = model_merge.predict([test_data_hastitle['txt'], test_data_hastitle['title'],
#                                            test_data_hastitle['target_entity'],
#                                            title_predict_hastitle_test,
#                                            txt_predict_hastitle_test,
#                                            entity_position_test])
#    
#    predict_hastitle_train = model_merge.predict([train_data_hastitle['txt'], train_data_hastitle['title'],
#                                                  train_data_hastitle['target_entity'],
#                                                  title_predict_hastitle, txt_predict_hastitle,
#                                                  entity_position_train])
#    
##%% 3.2 (处理没有title的)txt_predict，和最终模型混合----------
#    ## txt
##    K.clear_session()
#    K.clear_session()
#    ## merge model
#    model_merge_notitle = build_model_add_title_merge3(embedMatrix_notitle, learning_rate, 
#                                                       nb_words_notitle,
#                                                       max_length = train_data_notitle['txt'].shape[1],
#                                                       embedding_size = embedMatrix_notitle.shape[1],
#                                                       metric = f1)
#    
#    entity_position_train_notitle = get_target_entity_position2(train_data_notitle['target_entity'],
#                                                                train_data_notitle['txt'])
#
#    model_merge_notitle.fit([train_data_notitle['txt'],
#                             train_data_notitle['target_entity'],
#                             txt_predict_notitle,
#                             entity_position_train_notitle], 
#                            train_data_notitle['y_entity_negative'],
#                            batch_size=32,
#                            epochs= 5)
#    ## 预测测试集
#    entity_position_test_notitle = get_target_entity_position2(test_data_notitle['target_entity'],
#                                                                test_data_notitle['txt'])
#
#    predict_ignore_title = model_merge_notitle.predict([test_data_notitle['txt'],
#                                                        test_data_notitle['target_entity'],
#                                                        txt_predict_notitle_test,
#                                                        entity_position_test_notitle])
#    
#    predict_notitle = model_merge_notitle.predict([train_data_notitle['txt'],
#                                                   train_data_notitle['target_entity'],
#                                                   txt_predict_notitle,
#                                                   entity_position_train_notitle])
#    
    ## 3.3 保存预测结果
    with open(r'./predict_hastitle_word2vec.pkl', 'wb') as f:
        pickle.dump(title_predict_hastitle_test,f)
    with open(r'./predict_ignore_title_word2vec.pkl', 'wb') as f:
        pickle.dump(txt_predict_notitle_test,f)

    with open(r'./train_predict_hastitle_word2vec.pkl', 'wb') as f:
        pickle.dump(title_predict_hastitle,f)
    with open(r'./train_predict_ignore_title_word2vec.pkl', 'wb') as f:
        pickle.dump(txt_predict_notitle,f)