#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 14:18:13 2019

@author: chenjiannan
"""

from gensim.models import word2vec, KeyedVectors

def example_word2vec_with_pre_train():
    ''' 本函数为示例，通过预训练的模型训练词向量的方法,
    Returns：
        训练的模型
    '''
    #%% 1.case
    sentences = [['first', 'sentence'], ['second', 'sentence']]
    
    wv_from_text = word2vec.Word2Vec(size=10, min_count=1)  # 创建word2vec的结构
    # 往字典中加入新的词语,并构造哈夫曼树用于字典的查询
    # 初始化字典, 训练的时候,只会训练字典里有的词语对应的词向量
    wv_from_text.build_vocab(sentences)
    total_examples = wv_from_text.corpus_count  # 参与训练的语料数目
    
    # 训练第二个词向量
    sentences2 = [['third', 'sentence'], ['fourth', 'sentence']]
    model1 = word2vec.Word2Vec(sentences2, min_count=1, size=10)  # 会自动加入词向量的字典
    model1.wv.save_word2vec_format('test.txt')  # save the model
    
    # 混合两个模型
    wv_from_text.build_vocab([list(model1.wv.vocab.keys())], update=True)  # 加入新的字典的key
    # lockf:Use 1.0 to allow further training updates of merged vectors.
    # Merge in an input-hidden weight matrix loaded from the original C word2vec-tool format,
    # where it intersects with the current vocabulary.
    wv_from_text.intersect_word2vec_format("test.txt", binary=False, lockf=1.0)
    wv_from_text.train(sentences, total_examples=total_examples, epochs=wv_from_text.epochs)
    return wv_from_text

def word2vec_with_pre_train(pre_trained_word2vec_bin_file, output_word2vec):
    ''' 通过预训练的模型训练词向量的方法
    Args:
        pre_trained_word2vec_bin_file:读取的预训练模型，非二进制
        output_word2vec:输出的训练模型，二进制保存
    Returns：
        输出训练的模型到文件output_word2vec
    '''
    #%% 2.train own wordvec by pre-tained word vector
    # 使用Text8Corpus需要设置每个句子的最大长度max_sentence_length,大于该长度则认为是一个句子
    # sentences = word2vec.Text8Corpus('all_word_seg.txt', max_sentence_length=200)
    # 避免分句不当的影响,直接载入sentences
    f = open('all_word_seg.txt','r', encoding = 'UTF-8')
    sentences = f.readlines()
    sentences = [item[:-1].split(' ') for item in sentences]
    f.close()
    
    
    wv_from_text = word2vec.Word2Vec(size=200, min_count=1)
    wv_from_text.build_vocab(sentences)
    total_examples = wv_from_text.corpus_count
    
    model1 = KeyedVectors.load_word2vec_format(pre_trained_word2vec_bin_file, binary=False)
    
    wv_from_text.build_vocab([list(model1.wv.vocab.keys())], update=True)
    wv_from_text.intersect_word2vec_format(pre_trained_word2vec_bin_file, binary=False, lockf=1.0)
    wv_from_text.train(sentences, total_examples=total_examples, epochs=wv_from_text.epochs)
    wv_from_text.wv.save_word2vec_format(output_word2vec, binary=True)



if __name__ == '__main__':
    pre_trained_word2vec_bin_file = "Tencent_AILab_ChineseEmbedding.txt"
    output_word2vec = 'train_vec_byTencent_word.bin'
    word2vec_with_pre_train(pre_trained_word2vec_bin_file, output_word2vec)
    