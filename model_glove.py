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

if __name__ == '__main__':
    print('Loading data...')
#%% 1. loda train data from file
    ## 1.1 读取文件
    ### 读取嵌入矩阵---------------------------------------
    with open('word2idx_embedMatrix_glove.pkl', 'rb') as f:
        word2idx_notitle, embedMatrix_notitle = pickle.load(f)
    with open('word2idx_embedMatrix_glove.pkl', 'rb') as f:
        word2idx_hastitle, embedMatrix_hastitle = pickle.load(f)   
    # 有title的部分的字典中，词的数目
    nb_words_hastitle = len(word2idx_hastitle.keys()) + 1
    # 没有title的部分的字典中，词的数目
    nb_words_notitle = len(word2idx_notitle.keys()) + 1
    ### 读取训练集和测试集---------------------------------
    with open('train_data_model_glove.pkl', 'rb') as f:
        train_data_notitle = pickle.load(f)
    with open('train_data_model_hastitle_glove.pkl', 'rb') as f:
        train_data_hastitle = pickle.load(f)

#%% 2.load test data, 载入测试集数据，用于预测
    with open('test_data_model_glove.pkl', 'rb') as f:
        test_data_notitle = pickle.load(f)
    with open('test_data_model_hastitle_glove.pkl', 'rb') as f:
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
    
    ## 3.3 保存预测结果
    with open(r'./predict_hastitle_glove.pkl', 'wb') as f:
        pickle.dump(title_predict_hastitle_test,f)
    with open(r'./predict_ignore_title_glove.pkl', 'wb') as f:
        pickle.dump(txt_predict_notitle_test,f)
        
    with open(r'./train_predict_hastitle_glove.pkl', 'wb') as f:
        pickle.dump(title_predict_hastitle,f)
    with open(r'./train_predict_ignore_title_glove.pkl', 'wb') as f:
        pickle.dump(txt_predict_notitle,f)