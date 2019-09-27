# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 00:21:01 2019

@author: cjn
"""

import re
import pandas as pd
import numpy as np
from gensim.models import word2vec
import jieba

#  去掉中英文状态下的逗号、句号
def clearSen(comment):
    ''' 输入句子,输出清理了特殊字符后的句子
    '''
    if isinstance(comment, str):
        txt = re.sub(r'[\?]', '', comment) # 去除识别错误的问号?(英文字符)
        txt = re.sub(r'\[超话\]',' ',txt)  # 将[超话]变为一个空格
        txt = re.sub(r'&quot|&gt|&nbsp|…|<p>|<//p>|<\/a>|<a>|<\/p>|<strong>|'
                     +r'<\/strong>|<\/articlead>|<articlead>|< articlead>|IMG:\d|IMG \d|'
                     +r'<header>|<\/header>|<div>|<\/div>|<span>|<\/span>|'
                     +r'<br>|<\/br>|<article>|<\/article>',' ',txt)  # 将引号变为一个空格
        txt = re.sub(r'\.\.\.',' ',txt)  # 将省略号变为一个空格
        txt = re.sub(r'>>>',' ',txt)  # 将>>>变为一个空格
        txt = re.sub(r'(?<!http):',' ',txt)  # 删除前面不是http的冒号
 
        
        r = re.search(r'\d、\D',txt)  # 将1、2、之类变为一个空格(排除数字)
        if r:
            txt = re.sub(r.group(0),' ',txt)

        r = re.search(r'\d\.\D',txt)  # 将1. 2. 之类变为一个空格(排除数字)
        if r:
            txt = re.sub(r.group(0),' ',txt)
        # 将特殊字符变为一个空格
        txt = re.sub(r'[『』◆▲+▼「」@#《》【】，,。？！!“"”；;·\-：\[\]\/～~、丶 ()（）｜\|_丨→\{\}｛｝]',' ',txt)
        txt = re.sub(r'[―—]',' ',txt)  # 将特殊字符变为一个空格
        txt = re.sub(r'\s+',' ',txt)  # 将多个空格变为1个
        return txt.strip()
    else:
        return comment

def clear_entity(entity):
    ''' 输入句子,输出清理了特殊字符后的句子
    '''
    if isinstance(entity, str):
        txt = re.sub(r'[\?]', '', entity) # 去除识别错误的问号?(英文字符)
        return txt
    else:
        return entity


def fetch_entites(df, df_name, sep):
    ''' 输入关于entity的某一列，输出entity的set
    args:
        df:需要分析的dataframe
        df_name:需要提取entity的列名
        sep:分隔符
    '''
    all_entity = set()
    for entity_txt in df[df_name]:
        if isinstance(entity_txt, str):
            entities = set(entity_txt.split(sep))
            if len(entities) > 0:
                for entity in entities:
                    all_entity.add(entity.strip())
    return all_entity

def get_entity(entity_txt, sep):
    ''' 从entity字段中获取entity
    args:
        entity_txt:需要提取entity的字段值
        sep:分隔符
    '''
    all_entity = set()
    if isinstance(entity_txt, str):
        entities = set(entity_txt.split(sep))
        if len(entities) > 0:
            for entity in entities:
                all_entity.add(entity.strip())
    return all_entity

def seperate_txt(text):
    ''' file_path:输出分割结果的句子
    sep_column为需要使用jiaba进行分词的字段
    输入每个句子的分词结果到f文件
    '''
    if isinstance(text, str):
        sentence = []
        for item in jieba.cut(text):
            item_tmp = item.strip()
            if len(item_tmp) > 0:
                # 如果全为数字，则不加入sentence
                result = re.search('\d+', item_tmp)
                if result:
                    if result.group(0) == item_tmp:
                        continue
                sentence.append(item_tmp)
        if len(sentence) > 0:
            return  ' '.join(sentence)+'\n'
    else:
        return text
    
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
    #%%1.数据清洗
    train_file_path = './data/Train_Data.csv'
    data_train = pd.read_csv(train_file_path)
    data_train['title'] = data_train['title'].apply(clearSen)
    data_train['text'] = data_train['text'].apply(clearSen)
    data_train['key_entity'] = data_train['key_entity'].apply(clear_entity)
    data_train['entity'] = data_train['entity'].apply(clear_entity)
#    data_train.to_pickle('./Train_Data.pkl')
    
    test_file_path = './data/Test_Data.csv'
    data_test = pd.read_csv(test_file_path)
    data_test['title'] = data_test['title'].apply(clearSen)
    data_test['text'] = data_test['text'].apply(clearSen)
    data_test['entity'] = data_test['entity'].apply(clear_entity)
#    data_test.to_pickle('./Test_Data.pkl')
    
    #%%2.提取entities
    entities = fetch_entites(data_train, 'key_entity', ';')
    entities = entities.union(fetch_entites(data_train, 'entity', ';'))
    with open('financial_entity.txt', 'w', encoding = 'UTF-8') as f:
        for entity in entities:
            if len(entity) > 0:
                f.write(entity+' 999\n')
                
    #%%3.载入分词，调整分词
    jieba.load_userdict('financial_entity.txt')
    for j in range(10):
        for_suggest_enties = set()
        for i in data_train.index:
            title = data_train.loc[i, 'title']  # 当前title
            text = data_train.loc[i, 'text']  # 当前txt
            title_cut = ''
            text_cut = ''
            if isinstance(title, str):
                title_cut = '\\'.join(jieba.cut(title))
            if isinstance(text, str):
                text_cut = '\\'.join(jieba.cut(text))
            for entity in get_entity(data_train.loc[i, 'entity'], ';'):
                if len(title_cut) > 0:
                    if title.count(entity) != title_cut.count(entity):
                        for_suggest_enties.add(entity)
                if len(text_cut) > 0:
                    if text.count(entity) != text_cut.count(entity):
                        for_suggest_enties.add(entity)
        for entity in for_suggest_enties:
            if len(entity) > 0:
                jieba.suggest_freq(entity, tune=True)
        print('分割失败的entity的数目:{}'.format(len(for_suggest_enties)))
        
    #%%4.将分割后的结果保存在训练集中
    data_train['title_split'] = data_train['title'].apply(seperate_txt)
    data_train['txt_split'] = data_train['text'].apply(seperate_txt)
    data_train.to_pickle('./Train_Data.pkl')


    #%%5.加入测试集的entity
    entities = fetch_entites(data_test, 'entity', ';')
    with open('financial_entity_test.txt', 'w', encoding = 'UTF-8') as f:
        for entity in entities:
            if len(entity) > 0:
                f.write(entity+' 999\n')
    jieba.load_userdict('financial_entity_test.txt')

    #%%6.将分割后的结果保存在测试集中
    data_test['title_split'] = data_test['title'].apply(seperate_txt)
    data_test['txt_split'] = data_test['text'].apply(seperate_txt)
    data_test.to_pickle('./Test_Data.pkl')
    
    #%%7.输出分割的句子
#    data_train = pd.read_pickle('./Train_Data.pkl')
#    data_test = pd.read_pickle('./Test_Data.pkl')
    write_column_seg_in(data_train, 'title_split', './all_word_seg.txt')
    write_column_seg_in_add(data_train, 'txt_split', './all_word_seg.txt')
    write_column_seg_in_add(data_test, 'title_split', './all_word_seg.txt')
    write_column_seg_in_add(data_test, 'txt_split', './all_word_seg.txt')