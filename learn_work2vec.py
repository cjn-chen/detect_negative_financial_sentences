# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 18:04:45 2019
用于学习词向量
@author: cjn
"""
from gensim.models import word2vec

if __name__ == '__main__':
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
    
    
    