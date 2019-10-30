# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 16:59:08 2019

@author: cjn
"""
from mittens import GloVe
from my_utils import make_deepLearn_data
#from keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle
import pandas as pd
from keras.preprocessing.sequence import pad_sequences
from gensim.models import KeyedVectors

def generate_token(x, word2idx):
    result = []
    for item in x:
        result.append(word2idx.get(item, 0))
    return result

def make_deepLearn_data_entity(w):
    X_train_idx = word2idx.get(w, 0)
    return X_train_idx

def get_all_entities():
   ''' get all entities from sentence and entity field
   Args:
       entity_max_len:每个句子最多取几个词
   '''
   f = open('all_word_seg.txt','r', encoding = 'UTF-8')
   sentences = f.readlines()
   sentences = [item[:-1].split(' ') for item in sentences]
   f.close()
   entities_all = set()
   for sen in sentences:
       for item in sen:
           entities_all.add(item)
#   print(len(entities_all))

   f = open('financial_entity_test.txt','r', encoding = 'UTF-8')
   entities = f.readlines()
   entities = [item[:-4].strip() for item in entities]
   f.close()
   entities = set(entities)
   entities_all = entities_all.union(entities)
#   print(len(entities_all))

   f = open('financial_entity.txt','r', encoding = 'UTF-8')
   entities = f.readlines()
   entities = [item[:-4].strip() for item in entities]
   f.close()
   entities = set(entities)
   entities_all = entities_all.union(entities)
#   print(len(entities_all))
   return entities_all

def split_word(txt):
    if isinstance(txt, str):
        result = txt[:-1].split(' ')
    else:
        result = []
    return result

def split_word_entity(txt, sep=';'):
    if isinstance(txt, str):
        result = txt.split(sep)
    else:
        result = []
    return result

def deal_with_one_sentence(token_list, cooccurrence_matrix, window_size):
    ''' 输入一个已经经过tokenize的句子，句子由数字构成，输出经过修改的共现矩阵
    used in generate_co_occurrence
    Args:
        token_list:句子中的单词都被标记为数字
        cooccurrence_matrix:共现矩阵
        window_size:窗长
    '''
    half_window_size = (window_size - 1)/2
    end_index = len(token_list)-1
    if len(token_list) <= half_window_size:
        return
    for i in range(len(token_list)):
        left_index = int(max(i-half_window_size, 0))
        right_index = int(min(i+half_window_size, end_index))
        to_add_index = token_list[left_index:right_index+1]
        cooccurrence_matrix[token_list[i],to_add_index] += 1
    

def generate_co_occurrence(token_matrix, word_num, window_size = 5,):
    ''' 生成共现矩阵
    Args:
        token_matrix:每行为一个句子，句子中的单词都被标记为数字
        word_num:token的个数
        window_size:窗长
    '''
    try:
        assert window_size%2 == 1
    except:
        print('window_size must be odd number')
        raise ValueError
    process_i = 0
    index_i = 0
    cooccurrence_matrix = np.zeros((word_num+1, word_num+1))
    for token_list in token_matrix:
        index_i += 1
        if index_i//(token_matrix.shape[0]//10) > process_i:
            # 提示进度
            process_i += 1
            print('\r进度{}%'.format(round(index_i/token_matrix.shape[0]*100)), end='')
        deal_with_one_sentence(token_list, cooccurrence_matrix, window_size)
    return cooccurrence_matrix

def generate_count(entities_all, seg_path):
    entities_all_count = dict.fromkeys(entities_all,0)
    with open(seg_path,'r', encoding = 'UTF-8') as f:
        sentences = f.readlines()
        sentences = [item[:-1].split(' ') for item in sentences]
    for item in sentences:
        for item_j in item:
            if item_j in entities_all:
                entities_all_count[item_j] += 1
    return entities_all_count

def clear_entity(entities_all_count, limit_count = 1):
    entities_all_count_keys = set(entities_all_count.keys())
    print('entity清理前,清理频率低于{},且不属于识别实体的,数目{}'.format(
            limit_count,len(entities_all_count_keys)))
    entities_all = set()
    f = open('financial_entity_test.txt','r', encoding = 'UTF-8')
    entities = f.readlines()
    entities = [item[:-4].strip() for item in entities]
    f.close()
    entities = set(entities)
    entities_all = entities_all.union(entities)
#   print(len(entities_all))

    f = open('financial_entity.txt','r', encoding = 'UTF-8')
    entities = f.readlines()
    entities = [item[:-4].strip() for item in entities]
    f.close()
    entities = set(entities)
    entities_all = entities_all.union(entities)
    
    for key in entities_all_count.keys():
        if key not in entities_all and entities_all_count[key] < limit_count:
            entities_all_count_keys.remove(key)
    print('entity清理后，数目{}'.format(len(entities_all_count_keys)))
    return entities_all_count_keys

def merge_glove_word2vec_embedding(word2idx, embedMatrix, model):
    ''' 输入glove构造的矩阵，word2vec对应的model，输出添加了word2vec的向量的嵌入矩阵
    '''
    # 转化为index对应到word的字典
    idx2word = dict(zip(word2idx.values(), word2idx.keys()))
    # 构造新的embedding_matrix
    new_embedMatrix = np.zeros((embedMatrix.shape[0], embedMatrix.shape[1]+model.vector_size))
    for i in range(new_embedMatrix.shape[0]):
        if idx2word.__contains__(i):
            if model.__contains__(idx2word[i]):
                new_embedMatrix[i,embedMatrix.shape[1]:] = model[idx2word[i]]
    new_embedMatrix[:,:embedMatrix.shape[1]] = embedMatrix
    return new_embedMatrix

def generate_training_data(data_train_file, output_file, word2idx):
    ''' 生成tokenize后的数据
    Args:
        data_train_file:训练集文件
            output_file:输出的tokenize后的文件
    '''
    data_train = pd.read_pickle(data_train_file)
    x_train_txt0 = data_train.txt_split.apply(split_word)
    X_train_txt, _ = make_deepLearn_data(x_train_txt0, word2idx)
    
    x_train_title0 = data_train.title_split.apply(split_word)
    X_train_title, _ = make_deepLearn_data(x_train_title0, word2idx)
    
    x_entity = data_train.entity.apply(split_word_entity).apply(generate_token,args=(word2idx,))
    
    y_train = data_train.negative.values

    train_data = dict(zip(['txt', 'title', 'entity',
                           'y_train'], 
                          [X_train_txt, X_train_title, x_entity.values,
                           y_train]))
    with open(output_file, 'wb') as f:
        pickle.dump(train_data, f)
    
    shape_dic = {'txt_shape':X_train_txt.shape[1], 
            'title_shape':X_train_title.shape[1],}
    return shape_dic

def generate_test_data(data_test_file, output_file, shape_dic, word2idx):
    ''' 生成tokenize后的数据
    Args:
        data_test_file:test set文件
            output_file:输出的tokenize后的文件
            shape_dic:由generate_training_data生成的txt，title的shape
    '''
    data_test = pd.read_pickle(data_test_file)
    x_test_txt0 = data_test.txt_split.apply(split_word)
    X_test_txt, _ = make_deepLearn_data(x_test_txt0, word2idx)
    
    x_test_title0 = data_test.title_split.apply(split_word)
    X_test_title, _ = make_deepLearn_data(x_test_title0, word2idx)
    
    
    x_entity = data_test.entity.apply(split_word_entity).apply(generate_token,args=(word2idx,))
    
    # 保证test set的padding长度 和train set一致
    if shape_dic['txt_shape'] > X_test_txt.shape[1]:
        X_test_txt = pad_sequences(X_test_txt, shape_dic['txt_shape'], padding='post')
    else:
        X_test_txt = X_test_txt[:,:shape_dic['txt_shape']]
        
    if shape_dic['title_shape'] > X_test_title.shape[1]:
        X_test_title = pad_sequences(X_test_title, shape_dic['title_shape'], padding='post')
    else:
        X_test_title = X_test_title[:,:shape_dic['title_shape']]
    
    ## ouput file
    test_data = dict(zip(['txt', 'title', 'entity'], 
                          [X_test_txt, X_test_title, x_entity.values]))
    with open(output_file, 'wb') as f:
        pickle.dump(test_data, f)

if __name__ == '__main__':
#%% 1.训练Glove并输出
    dim_n = 200
    itter_n = 1000
    # 读取所有实体
    entities_all = get_all_entities()
    # 读取所有句子，包括测试集和训练集
    with open('all_word_seg.txt','r', encoding = 'UTF-8') as f:
        sentences = f.readlines()
        sentences = [item[:-1].split(' ') for item in sentences]
    
    ## 清除低频的entity
    entities_all_count = generate_count(entities_all, 'all_word_seg.txt')
    entities_all_count_keys = clear_entity(entities_all_count,limit_count=3)
    
    ## 从1开始，留一位给UNK，从1开始标
    word2idx = dict(zip(entities_all_count_keys, range(1, len(entities_all_count_keys)+1)))

    # 对句子进行tokenize，生成token矩阵
    sentences_tokens,_ = make_deepLearn_data(sentences, word2idx)
    # 生成共现矩阵
    cooccurrence_matrix = generate_co_occurrence(sentences_tokens, len(word2idx.keys()),
                                                 window_size = 5,)
    # 训练glove并输出文件
    glove_model = GloVe(n = dim_n, max_iter = itter_n)
    embedMatrix = glove_model.fit(cooccurrence_matrix)
    
    
    print('load word2vec model...')
    model = KeyedVectors.load_word2vec_format('train_vec_byTencent_word.bin', binary=True)
    
    print('build embedding matrix...')
    new_embedMatrix = merge_glove_word2vec_embedding(word2idx, embedMatrix, model)

    
    with open('word2idx_embedMatrix_glove_word2vec.pkl', 'wb') as f:
        pickle.dump([word2idx, new_embedMatrix], f)
    
    
#%% 3.生成训练集和测试集
    ## no title
    print('produce training set...')
    data_train_file = 'Train_Data.pkl'
    output_file = 'train_data_model_glove_word2vec.pkl'
    shape_dic = generate_training_data(data_train_file, output_file,word2idx)
    
    print('produce test set...')
    data_test_file = 'Test_Data.pkl'
    output_file = 'test_data_model_glove_word2vec.pkl'
    generate_test_data(data_test_file, output_file, shape_dic,word2idx)
    
    ## have title
    print('produce training set (has tille)...')
    data_train_file = 'Train_Data_hastitle.pkl'
    output_file = 'train_data_model_hastitle_glove_word2vec.pkl'
    shape_dic = generate_training_data(data_train_file, output_file, word2idx)
    
    print('produce test set (has tille)...')
    data_test_file = 'Test_Data_hastitle.pkl'
    output_file = 'test_data_model_hastitle_glove_word2vec.pkl'
    generate_test_data(data_test_file, output_file, shape_dic, word2idx)
    
    
