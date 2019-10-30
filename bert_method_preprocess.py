#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 10:59:45 2019

@author: chenjiannan
"""
from keras_bert import load_trained_model_from_checkpoint, Tokenizer
import pandas as pd
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle

def get_all_entities():
   ''' get all entities from sentence and entity field
   '''
   f = open('all_word_seg.txt','r', encoding = 'UTF-8')
   sentences = f.readlines()
   sentences = [item[:-1].split(' ')[:100] for item in sentences]
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

class OurTokenizer(Tokenizer):
    ''' 自定义自己的tokenize
    '''
    def _tokenize(self, text):
        R = []
        for c in text:
            if c in self._token_dict:
                R.append(c)
            elif self._is_space(c):
                R.append('[unused1]') # space类用未经训练的[unused1]表示
            else:
                R.append('[UNK]') # 剩余的字符是[UNK]
        return R

def generate_encode(split_sentences, tokenizer):
    ''' 生成token编码
    Args:
        txts:分好词的结果,按照空格分
        tokenizer:keras_bert中的Tokenizer
    '''
    x_indices = []
    x_segments = []
    for txt_split in split_sentences:
        indices, segments = tokenizer.encode(txt_split[:-1].split(' '))
        x_indices.append(indices)
        x_segments.append(segments)
    return x_indices,x_segments

def make_deepLearn_data_bert(sentenList, word2idx, maxlen = -1):
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
    if maxlen == -1:
        maxlen = 0
        for i in sentenList:
            if len(i)>maxlen:
                maxlen = len(i)

    
    X_train_idx = [[word2idx.get(w, word2idx['[UNK]']) for w in sen[:maxlen]] for sen in sentenList]
    X_train_idx = np.array(pad_sequences(X_train_idx, maxlen, padding='post'))  # 必须是np.array()类型
    return X_train_idx, maxlen

def split_word_bert(txt):
    if isinstance(txt, str):
        result = ['[SEP]']+txt[:-1].split(' ')+['[SEP]']
    else:
        result = []
    return result

def make_deepLearn_data_entity_bert(w, word2idx):
    X_train_idx = word2idx.get(w, word2idx['[UNK]'])
    return X_train_idx

def make_deepLearn_data_entity(w, word2idx):
    X_train_idx = word2idx.get(w, word2idx['[UNK]'])
    return X_train_idx

def get_token_dict():
    ''' 根据训练集和测试集中的entity和txt,title添加entity到token字典中
    Returns:返回token dict()
    '''
    entities_all = get_all_entities()
    if '' in entities_all:
        entities_all.remove('')
    token_dict = {
        '[CLS]': 0,
        '[SEP]': 1,
        '[UNK]': 2,
        '[unused1]':3
    }
    entities_all_dict = dict(zip(entities_all, 
                                 range(len(token_dict),
                                       len(entities_all)+len(token_dict))))
    token_dict = dict(**token_dict,
                      **entities_all_dict)
    return token_dict

def generate_training_data(data_train_file, output_file, token_dict):
    ''' 生成tokenize后的数据
    Args:
        data_train_file:训练集文件
            output_file:输出的tokenize后的文件
            token_dict:token字典
    Returns:
        shape_dic:由generate_training_data生成的txt，title, 的shape
    '''
    data_train = pd.read_pickle(data_train_file)
    x_train_txt0 = data_train.txt_split.apply(split_word_bert)
    X_train_txt, _ = make_deepLearn_data_bert(x_train_txt0, token_dict)
    #temp=data_train.txt_split.apply(lambda x: x.count(' '))
    #np.percentile(temp,95)
    maxlen_title = -1 #不指定最大长度
    x_train_title0 = data_train.title_split.apply(split_word_bert)
    X_train_title, maxlen_title = make_deepLearn_data_bert(x_train_title0, token_dict, maxlen = maxlen_title)
    
    target_entity_train = data_train.target_entity.apply(make_deepLearn_data_entity,args=(token_dict,))
    y_train = data_train.negative.values
    y_entity_negative = data_train.entity_negative.values

    train_data = dict(zip(['txt', 'title', 'target_entity',
                           'y_entity_negative', 'y_train'], 
                          [X_train_txt, X_train_title, target_entity_train,
                           y_entity_negative, y_train]))
    with open(output_file, 'wb') as f:
        pickle.dump(train_data, f)
    shape_dic = {'txt_shape':X_train_txt.shape[1], 
                 'title_shape':X_train_title.shape[1]}
    return shape_dic

def generate_test_data(data_test_file, output_file, token_dict, shape_dic):
    ''' 生成tokenize后的数据
    Args:
        data_test_file:test set文件
            output_file:输出的tokenize后的文件
            shape_dic:由generate_training_data生成的txt，title的shape
    '''
    ## 保证test set的padding长度 和train set一致
    data_test = pd.read_pickle(data_test_file)
    x_test_txt0 = data_test.txt_split.apply(split_word_bert)
    X_test_txt, _ = make_deepLearn_data_bert(x_test_txt0, token_dict, maxlen = -1)

    x_test_title0 = data_test.title_split.apply(split_word_bert)
    X_test_title, _ = make_deepLearn_data_bert(x_test_title0, token_dict, maxlen = -1)
    
    target_entity_test = data_test.target_entity.apply(make_deepLearn_data_entity,args=(token_dict,))
    
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
    test_data = dict(zip(['txt', 'title', 'target_entity'], 
                          [X_test_txt, X_test_title, target_entity_test]))
    with open(output_file, 'wb') as f:
        pickle.dump(test_data, f)


if __name__ == '__main__':
    ## 1.Token
    ## 1.1构造token字典
    token_dict = get_token_dict()

    ## 1.2进行tokenizer
    ### no title
    print('produce training set (no tille)...')
    data_train_file = 'Train_Data.pkl'
    output_file = 'train_data_model_bert.pkl'
    shape_dic = generate_training_data(data_train_file, output_file, token_dict)
    
    print('produce test set (no tille)...')
    data_test_file = 'Test_Data.pkl'
    output_file = 'test_data_model_bert.pkl'
    generate_test_data(data_test_file, output_file, token_dict, shape_dic)

    ### has title
    print('produce training set (has tille)...')
    data_train_file = 'Train_Data_hastitle.pkl'
    output_file = 'train_data_model_hastitle_bert.pkl'
    shape_dic = generate_training_data(data_train_file, output_file, token_dict)
    
    print('produce test set (has tille)...')
    data_test_file = 'Test_Data_hastitle.pkl'
    output_file = 'test_data_model_hastitle_bert.pkl'
    generate_test_data(data_test_file, output_file, token_dict, shape_dic)