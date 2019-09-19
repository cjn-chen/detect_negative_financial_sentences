# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 18:04:45 2019
用于学习词向量
@author: cjn
"""
import pandas as pd
from gensim.models import word2vec

def write_seg_txt(txt):
    if isinstance(txt, str):
        return txt
    else:
        return ''

def write_column_seg_in(df, column, output_file_path):
    ''' 将df中的某一列输出到文件中
    '''
    file = open(output_file_path, 'w', encoding = 'UTF-8')
    for title_item in data_train[column]:
        tmp_txt = write_seg_txt(title_item)
        if len(tmp_txt) > 0 and isinstance(tmp_txt, str):
            file.write(tmp_txt)

def write_column_seg_in_add(df, column, output_file_path):
    ''' 将df中的某一列输出到文件中
    '''
    file = open(output_file_path, 'w+', encoding = 'UTF-8')
    for title_item in data_train[column]:
        tmp_txt = write_seg_txt(title_item)
        if len(tmp_txt) > 0 and isinstance(tmp_txt, str):
            file.write(tmp_txt)

if __name__ == '__main__':
    #%%1.载入数据
    data_train = pd.read_pickle('./Train_Data_sen_vec.pkl')
    data_test = pd.read_pickle('./Test_Data_sen_vec.pkl')
        
    #%%2.输出分割的句子
    write_column_seg_in(data_train, 'title_split', './all_word_seg.txt')
    write_column_seg_in_add(data_train, 'txt_split', './all_word_seg.txt')
    write_column_seg_in_add(data_test, 'title_split', './all_word_seg.txt')
    write_column_seg_in_add(data_test, 'txt_split', './all_word_seg.txt')
    
    #%%3.训练词向量    
    # 使用Text8Corpus需要设置每个句子的最大长度max_sentence_length,大于该长度则认为是一个句子
    # sentences = word2vec.Text8Corpus('all_word_seg.txt', max_sentence_length=200)
    # 避免分句不当的影响,直接载入sentences
    f = open('all_word_seg.txt','r', encoding = 'UTF-8')
    sentences = f.readlines()
    sentences = [item[:-1].split(' ') for item in sentences]
    f.close()
    
    model = word2vec.Word2Vec(sentences, min_count=1, size = 300)
    model.wv.save_word2vec_format('train_vec_by_train_txt.word')
    
    
    