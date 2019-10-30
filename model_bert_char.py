from keras_bert import load_trained_model_from_checkpoint
import pickle
from my_utils import f1
import numpy as np
from keras.layers import Input, SpatialDropout1D, Bidirectional, \
                         GlobalMaxPooling1D, CuDNNLSTM, CuDNNGRU, Concatenate,\
                         Dense, GlobalAveragePooling1D, Lambda\
                         #Multiply, ,  BatchNormalization, Dropout
from keras.models import Model
from keras import optimizers
from keras.preprocessing.sequence import pad_sequences
import keras.backend as K
#from attention_keras import Attention, TrainablePositionEmbedding

config_path = './chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = './chinese_L-12_H-768_A-12/bert_model.ckpt'

def build_model_title_txt(learning_rate, max_length=55,
                          max_length_txt = 100,
                          metric = f1, batch_size=100):
    '''
    根据预训练的嵌入矩阵，返回神经网络的模型，返回模型还需要调用model.fit模块
    Args:
        embedding_matrix:嵌入矩阵,每行为一个单词，每列为其中一个维度
        nb_words:词汇表大小，设置为出现过的词汇数目+1，空的位置留给OOV(out of vocabulary),
        max_length:
    '''
    # 预训练模型
    bert_model = load_trained_model_from_checkpoint(config_path, 
                                                    checkpoint_path,
                                                    seq_len=None)
    for i in range(40):
        bert_model.layers[-i].trainable
    
    inp = Input(shape=(max_length,))  # 定义输入
    inp2 = Input(shape=(max_length,))  # 定义输入
    inp_txt = Input(shape=(max_length_txt,))
    inp_txt2 = Input(shape=(max_length_txt,))
    # title
    x = bert_model([inp, inp2])
    x = Lambda(lambda x:x)(x)
    x = SpatialDropout1D(0.3)(x)  # 对某一个维度进行dropout,embedding中的某一列
    x1 = Bidirectional(CuDNNLSTM(128, return_sequences=True))(x)  # 使用GPU加速的LSTM
    x2 = Bidirectional(CuDNNGRU(64, return_sequences=True))(x1)  # 使用GPU加速的GRU
    max_pool1 = GlobalMaxPooling1D()(x1)  #对于时序数据的全局最大池化，
    max_pool2 = GlobalMaxPooling1D()(x2)  #对于时序数据的全局最大池化。
    avg_pool1 = GlobalAveragePooling1D()(x1)
    avg_pool2 = GlobalAveragePooling1D()(x2)
    # txt
    x_txt = bert_model([inp_txt, inp_txt2])
    x_txt = Lambda(lambda x:x)(x_txt)
    x_txt = SpatialDropout1D(0.3)(x_txt)  # 对某一个维度进行dropout,embedding中的某一列
    x1_txt = Bidirectional(CuDNNLSTM(128, return_sequences=True))(x_txt)  # 使用GPU加速的LSTM
    x2_txt = Bidirectional(CuDNNGRU(64, return_sequences=True))(x1_txt)  # 使用GPU加速的GRU
    max_pool1_txt = GlobalMaxPooling1D()(x1_txt)  #对于时序数据的全局最大池化，
    max_pool2_txt = GlobalMaxPooling1D()(x2_txt)  #对于时序数据的全局最大池化。
    avg_pool1_txt = GlobalAveragePooling1D()(x1_txt)
    avg_pool2_txt = GlobalAveragePooling1D()(x2_txt)
    
    conc = Concatenate()([max_pool1, max_pool2,avg_pool1,avg_pool2,
                          max_pool1_txt, max_pool2_txt, avg_pool1_txt, avg_pool2_txt,],)  # 合并两层
    
    predictions = Dense(1, activation='sigmoid')(conc)

    model = Model(inputs=[inp,inp2,inp_txt,inp_txt2], outputs=predictions)
    adam = optimizers.Adam(lr=learning_rate)
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=[metric])
    return model

def build_model(learning_rate, 
                max_length=55, metric = f1,):
    '''
    根据预训练的嵌入矩阵，返回神经网络的模型，返回模型还需要调用model.fit模块
    Args:
        embedding_matrix:嵌入矩阵,每行为一个单词，每列为其中一个维度
        max_length:
    '''
    bert_model = load_trained_model_from_checkpoint(config_path, 
                                                    checkpoint_path,
                                                    seq_len=None)
    for i in range(40):
        bert_model.layers[-i].trainable
    inp = Input(shape=(max_length,))  # 定义输入
    inp2 = Input(shape=(max_length,))  # 定义输入

    x = bert_model([inp, inp2])
    x = Lambda(lambda x:x)(x)
    x = SpatialDropout1D(0.3)(x)  # 对某一个维度进行dropout,embedding中的某一列
    x1 = Bidirectional(CuDNNLSTM(256, return_sequences=True))(x)  # 使用GPU加速的LSTM
    x2 = Bidirectional(CuDNNGRU(128, return_sequences=True))(x1)  # 使用GPU加速的GRU
    max_pool1 = GlobalMaxPooling1D()(x1)  #对于时序数据的全局最大池化，
    max_pool2 = GlobalMaxPooling1D()(x2)  #对于时序数据的全局最大池化。
    avg_pool1 = GlobalAveragePooling1D()(x1)
    avg_pool2 = GlobalAveragePooling1D()(x2)
    conc = Concatenate()([max_pool1, max_pool2,avg_pool1,avg_pool2])  # 合并两层
    predictions = Dense(1, activation='sigmoid')(conc)
    model = Model(inputs=[inp, inp2], outputs=predictions)
    adam = optimizers.Adam(lr=learning_rate)
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=[metric])
    return model

if __name__ == '__main__':
#%% 1. loda train data from file
    ## 1.1 读取文件
    ### 读取训练集和测试集---------------------------------
    with open('train_data_model_bert_char.pkl', 'rb') as f:
        train_data_notitle = pickle.load(f)
    with open('train_data_model_hastitle_bert_char.pkl', 'rb') as f:
        train_data_hastitle = pickle.load(f)
    
#%% 2.load test data, 载入测试集数据，用于预测
    with open('test_data_model_bert_char.pkl', 'rb') as f:
        test_data_notitle = pickle.load(f)
    with open('test_data_model_hastitle_bert_char.pkl', 'rb') as f:
        test_data_hastitle = pickle.load(f)
#%% 3.train the model and predict data
#%% 3.1模型的超参数
    learning_rate = 0.01
    # 选择95%的txt的长度进行截断
    all_txt_len = np.append((train_data_hastitle['txt']>0).sum(axis=1),(train_data_notitle['txt']>0).sum(axis=1))
    cut_shape_txt = min(int(np.percentile(all_txt_len,93)),500)

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
    model_title = build_model_title_txt(learning_rate,
                                        max_length = train_data_hastitle['title'].shape[1],
                                        max_length_txt = train_data_hastitle['txt'].shape[1],
                                        metric = f1,
                                        batch_size = train_data_hastitle['title'].shape[0])

    model_title.fit([train_data_hastitle['title'],
                     np.zeros(train_data_hastitle['title'].shape),
                     train_data_hastitle['txt'],
                     np.zeros(train_data_hastitle['txt'].shape)],
                    train_data_hastitle['y_train'],
                    batch_size = 32,
                    epochs = 3,
                    )
    
    title_predict_hastitle = model_title.predict([train_data_hastitle['title'],
                                                  np.zeros(train_data_hastitle['title'].shape),
                                                  train_data_hastitle['txt'],
                                                  np.zeros(train_data_hastitle['txt'].shape)])
    ### predict test set
    title_predict_hastitle_test = model_title.predict([test_data_hastitle['title'],
                                                       np.zeros(test_data_hastitle['title'].shape),
                                                       test_data_hastitle['txt'],
                                                       np.zeros(test_data_hastitle['txt'].shape),])
    K.clear_session()
    ## txt-----------------------------------------------------------------
    model_txt_notitle = build_model(learning_rate,
                                    max_length = train_data_notitle['txt'].shape[1],)
    
    model_txt_notitle.fit([train_data_notitle['txt'],
                           np.zeros(train_data_notitle['txt'].shape)],
                           train_data_notitle['y_train'],
                           batch_size = 32,
                           epochs = 2,)
    txt_predict_notitle = model_txt_notitle.predict([train_data_notitle['txt'],
                                                     np.zeros(train_data_notitle['txt'].shape)])
    
    ### predict test set
    txt_predict_notitle_test = model_txt_notitle.predict([test_data_notitle['txt'],
                                                           np.zeros(test_data_notitle['txt'].shape)])
    
    K.clear_session()
    
    ## 3.3 保存预测结果
    with open(r'./predict_hastitle_bert.pkl', 'wb') as f:
        pickle.dump(title_predict_hastitle_test,f)
    with open(r'./predict_ignore_title_bert.pkl', 'wb') as f:
        pickle.dump(txt_predict_notitle_test,f)
        
    with open(r'./train_predict_hastitle_bert.pkl', 'wb') as f:
        pickle.dump(title_predict_hastitle,f)
    with open(r'./train_predict_ignore_title_bert.pkl', 'wb') as f:
        pickle.dump(txt_predict_notitle,f)

    
    