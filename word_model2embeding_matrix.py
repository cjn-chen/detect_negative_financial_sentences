# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 13:33:52 2019
本文件的输入用于Keras中的Embedding函数使用
Embedding(nb_words, embedding_size, weights=[embedding_matrix], trainable=False)(inp)
@author: cjn
"""
from keras.preprocessing import sequence
import numpy as np
from gensim.models import KeyedVectors
import pickle
import pandas as pd

def build_word2idx_embedMatrix(model):
    ''' 输入gensim训练的词向量模型，输出{'词语'：下标}字典和嵌入矩阵
    注意：word2idx中的下标对应与embedMatrix中的行，比如word2idx['国外']为991，
    embedMatrix[991,:]为'国外'对应的词向量
    Args:
        model:经过gensim模型训练的结果
    Returns:
        word2idx:名词到index的映射的字典，比如 {'国外': 991,'网': 992, '贷网': 993,  ...}
        embedMatrix:每一行为一个单词，每一列为一个维度
    '''
    # 存储word2vec中所有向量的数组，留意其中多一位，词向量全为0，用于padding
    embedMatrix = np.zeros((len(model.wv.vocab.keys()) + 1, model.vector_size))
    # 构造词语到index的字典
    word2idx = {}  # 可以在word2idx中保留一个停词位,比如word2idx['_stop_word'] = 0
    word2idx = dict(zip(model.wv.vocab.keys(), range(1, len(model.wv.vocab.keys())+1)))
    # 构建
    embedMatrix[1:,:] = model.wv.__getitem__(word2idx.keys())
    return word2idx, embedMatrix

def build_word2idx_embedMatrix_2(model, entities_all):
    ''' 输入gensim训练的词向量模型，输出{'词语'：下标}字典和嵌入矩阵
    注意：word2idx中的下标对应与embedMatrix中的行，比如word2idx['国外']为991，
    embedMatrix[991,:]为'国外'对应的词向量
    Args:
        model:经过gensim模型训练的结果
        entities_all:所有句子以及标注的实体
    Returns:
        word2idx:名词到index的映射的字典，比如 {'国外': 991,'网': 992, '贷网': 993,  ...}
        embedMatrix:每一行为一个单词，每一列为一个维度
    '''
    word_set = set(model.wv.vocab.keys())
    word_interset = word_set.intersection(entities_all)
    # 存储word2vec中所有向量的数组，留意其中多一位，词向量全为0，用于padding
    embedMatrix = np.zeros((len(word_interset) + 1, model.vector_size))
    # 构造词语到index的字典
    word2idx = {}  # 可以在word2idx中保留一个停词位,比如word2idx['_stop_word'] = 0
    ## 把index为0的留给padding
    word2idx = dict(zip(word_interset, range(1, len(word_interset)+1)))
    # 构建
    embedMatrix[1:,:] = model.wv.__getitem__(word2idx.keys())
    return word2idx, embedMatrix

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
    # 确定句子大最大长度
    maxlen = 0
    for i in sentenList:
        if len(i)>maxlen:
            maxlen = len(i)
    
    X_train_idx = [[word2idx.get(w, 0) for w in sen] for sen in sentenList]
    X_train_idx = np.array(sequence.pad_sequences(X_train_idx, maxlen, padding='post'))  # 必须是np.array()类型
    return X_train_idx, maxlen

def split_word(txt):
    if isinstance(txt, str):
        result = txt[:-1].split(' ')
    else:
        result = []
    return result

if __name__ == '__main__':
#%% 1.生成嵌入矩阵,单词的字典
    model = KeyedVectors.load_word2vec_format('train_vec_byTencent_word.bin', binary=True)
    
    f = open('all_word_seg.txt','r', encoding = 'UTF-8')
    sentences = f.readlines()
    sentences = [item[:-1].split(' ') for item in sentences]
    f.close()
    entities_all = set()
    for sen in sentences:
        for item in sen:
            entities_all.add(item)
    print(len(entities_all))
    
    f = open('financial_entity_test.txt','r', encoding = 'UTF-8')
    entities = f.readlines()
    entities = [item[:-1].split(' ')[0] for item in entities]
    f.close()
    entities = set(entities)
    entities_all = entities_all.union(entities)
    print(len(entities_all))
    
    f = open('financial_entity.txt','r', encoding = 'UTF-8')
    entities = f.readlines()
    entities = [item[:-1].split(' ')[0] for item in entities]
    f.close()
    entities = set(entities)
    entities_all = entities_all.union(entities)
    print(len(entities_all))
    
    word2idx, embedMatrix = build_word2idx_embedMatrix_2(model, entities_all)
    with open('word2idx_embedMatrix.pkl', 'wb') as f:
        pickle.dump([word2idx, embedMatrix], f)
    
#%% 2.生成训练集和测试集
    ## train set
    data_train = pd.read_pickle('Train_Data.pkl')
    x_train_txt0 = data_train.txt_split.apply(split_word)
    y_train = data_train.negative.values
    X_train_txt, X_train_txt_max_len = make_deepLearn_data(x_train_txt0, word2idx)
    
    x_train_title0 = data_train.title.apply(split_word)
    X_train_title, X_train_title_max_len = make_deepLearn_data(x_train_title0, word2idx)

    train_data = dict(zip(['X_train_txt','X_train_txt_max_len',
                           'X_train_title','X_train_title_max_len', 'y_train'], 
                          [X_train_txt, X_train_txt_max_len,
                           X_train_title, X_train_title_max_len, y_train]))
    with open('train_data.pkl', 'wb') as f:
        pickle.dump(train_data, f)
    ## test set
    data_test = pd.read_pickle('Test_Data.pkl')
    x_test_txt0 = data_test.txt_split.apply(split_word)
    X_test_txt, X_test_txt_max_len = make_deepLearn_data(x_test_txt0, word2idx)
    
    x_test_title0 = data_test.title.apply(split_word)
    X_test_title, X_test_title_max_len = make_deepLearn_data(x_test_title0, word2idx)
    
    test_data = dict(zip(['X_test_txt','X_test_txt_max_len',
                           'X_test_title','X_test_title_max_len',], 
                          [X_test_txt, X_test_txt_max_len,
                           X_test_title, X_test_title_max_len,]))
    with open('test_data.pkl', 'wb') as f:
        pickle.dump(test_data, f)


