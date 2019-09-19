# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 11:40:09 2019

@author: cjn
"""
file = open('I:\MyDownloads\sgns.financial.word','r',encoding='UTF-8')
from gensim.models import KeyedVectors
wv_from_text = KeyedVectors.load_word2vec_format(file, binary=False)
