# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 23:40:32 2019

@author: cjn
"""
import numpy as np
import pandas as pd

def generate_sentences_vec(sentences_split, word2vec, vec_dim):
    ''' 将分词后的句子转化为句子向量,直接将所有词向量相加
    '''
    if isinstance(sentences_split, str):
        sentences = np.zeros((vec_dim))
        for entity in sentences_split:
            if word2vec.wv.__contains__(entity):
                sentences += word2vec.wv.__getitem__(entity)
        return sentences
    else:
        return np.zeros((vec_dim))

i = 0
j = 0
miss_entity = set()

def generate_entity_vec(entity_item, word2vec, vec_dim):
    ''' 将实体赋予词向量
    '''
    global i
    global j
    global miss_entity
    if isinstance(entity_item, str):
        sentences = []
        sentences_split = entity_item.split(';')
        for entity in sentences_split:
            j += 1
            if word2vec.wv.__contains__(entity):
                sentences.append(word2vec.wv.__getitem__(entity))
            else:
                i+=1
                miss_entity.add(entity)
                sentences.append(np.zeros((vec_dim)))
        if len(sentences) > 0:
            return sentences
        else:
            return []
    else:
        return []
    
    

#%% 1.载入词向量,同时将新的词训练进词向量
#file = open('I:\MyDownloads\sgns.financial.word','r',encoding='UTF-8')
        
#file = open('./Tencent_AILab_ChineseEmbedding.txt','r',encoding='UTF-8')   
#from gensim.models import KeyedVectors
#wv_from_text = KeyedVectors.load_word2vec_format(file, binary=False)
#vec_dim = wv_from_text['a'].shape[0]  # 获取维数


from gensim.models import KeyedVectors, word2vec

f = open('all_word_seg.txt','r', encoding = 'UTF-8')
sentences = f.readlines()
sentences = [item[:-1].split(' ') for item in sentences]
f.close()


wv_from_text = word2vec.Word2Vec(size=200, min_count=1)
wv_from_text.build_vocab(sentences)
total_examples = wv_from_text.corpus_count

model1 = KeyedVectors.load_word2vec_format("Tencent_AILab_ChineseEmbedding.txt", binary=False)

wv_from_text.build_vocab([list(model1.wv.vocab.keys())], update=True)
wv_from_text.intersect_word2vec_format("Tencent_AILab_ChineseEmbedding.txt", binary=False, lockf=1.0)
wv_from_text.train(sentences, total_examples=total_examples, epochs=wv_from_text.epochs)

vec_dim = wv_from_text['a'].shape[0]  # 获取维数


#%% 2.对输入的句子(已经经过分词)中的每个entity 进行赋予词向量
data_train = pd.read_pickle('./Train_Data.pkl')
data_test = pd.read_pickle('./Test_Data.pkl')

##
data_train['entity_vec'] = data_train['entity'].apply(generate_entity_vec, args = (wv_from_text,vec_dim,))
data_train['key_entity_vec'] = data_train['key_entity'].apply(generate_entity_vec, args = (wv_from_text,vec_dim,))

data_train['txt_sentence_vec'] = data_train['txt_split'].apply(generate_sentences_vec, args = (wv_from_text,vec_dim,))
data_train['title_sentence_vec'] = data_train['title_split'].apply(generate_sentences_vec, args = (wv_from_text,vec_dim,))

##
data_test['entity_vec'] = data_test['entity'].apply(generate_entity_vec, args = (wv_from_text,vec_dim,))

data_test['txt_sentence_vec'] = data_test['txt_split'].apply(generate_sentences_vec, args = (wv_from_text,vec_dim,))
data_test['title_sentence_vec'] = data_test['title_split'].apply(generate_sentences_vec, args = (wv_from_text,vec_dim,))



#%% 3.输出整理后的训练样本和测试样本
data_train.to_pickle('./Train_Data_sen_vec_Tencent_all.pkl')
data_test.to_pickle('./Test_Data_sen_Tencent_all.pkl')
