#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用户聚合所有模型的预测结果
Created on Tue Oct 22 16:54:15 2019

@author: chenjiannan
"""
import pandas as pd
import numpy as np
from fuzzywuzzy import fuzz
from sklearn.linear_model import LogisticRegression
from data_preprocess import fetch_entites

has_title_predcit_files = ['predict_hastitle_glove.pkl',
                           'predict_hastitle_glove_word2vec.pkl',
                           'predict_hastitle_word2vec.pkl',
                           'predict_hastitle_bert.pkl',]
ignore_title_predcit_files = ['predict_ignore_title_glove.pkl',
                              'predict_ignore_title_glove_word2vec.pkl',
                              'predict_ignore_title_word2vec.pkl',
                              'predict_ignore_title_bert.pkl',]

train_has_title_predcit_files = ['train_predict_hastitle_glove.pkl',
                                 'train_predict_hastitle_glove_word2vec.pkl',
                                 'train_predict_hastitle_word2vec.pkl',
                                 'train_predict_hastitle_bert.pkl',]
train_ignore_title_predcit_files = ['train_predict_ignore_title_glove.pkl',
                                    'train_predict_ignore_title_glove_word2vec.pkl',
                                    'train_predict_ignore_title_word2vec.pkl',
                                    'train_predict_ignore_title_bert.pkl',]
#
data_train_file = 'Train_Data.pkl'
data_test_file = 'Test_Data.pkl'
data_train_file_hastitle = 'Train_Data_hastitle.pkl'
data_test_file_hastitle = 'Test_Data_hastitle.pkl'

def get_data(data_train, data_test, data_train_hastitle, data_test_hastitle,):
    for file in train_has_title_predcit_files:
        method_name = file.replace('train_predict_hastitle_','').replace('.pkl','')
        data_train_hastitle[method_name] = pd.read_pickle(file).reshape(-1)
    
    for file in train_ignore_title_predcit_files:
        method_name = file.replace('train_predict_ignore_title_','').replace('.pkl','')
        data_train[method_name] = pd.read_pickle(file).reshape(-1)
    
    for file in has_title_predcit_files:
        method_name = file.replace('predict_hastitle_','').replace('.pkl','')
        data_test_hastitle[method_name] = pd.read_pickle(file).reshape(-1)
    
    for file in ignore_title_predcit_files:
        method_name = file.replace('predict_ignore_title_','').replace('.pkl','')
        data_test[method_name] = pd.read_pickle(file).reshape(-1)
        
def generate_ensemble_lg(x, predict_columns, threshold):
    '''
    '''
    if ((x['predict_logistic'] > 0.5+threshold)
        or (x['predict_logistic'] < 0.5-threshold)):
        return x['predict_logistic']
    predict_values = x[predict_columns].values
    abs_predict_value = np.abs(predict_values-0.5)
    max_value = max(abs_predict_value)
    if max_value > threshold:
        index = np.argmax(abs_predict_value)
        result = predict_values[index]
    else:
        result = x['predict_logistic']
    return result

def generate_predict_logistic(df,predict_columns):
    '''
    '''
    clf = LogisticRegression(random_state=0, solver='lbfgs',
                             multi_class='multinomial').fit(df[predict_columns].values, 
                                                            df['negative'])
    df.loc[:,'predict_logistic'] = clf.predict_proba(df[predict_columns].values)[:,1]
    print(sum(df['negative']==(df['predict_logistic']>0.5))/df.shape[0])
    return clf

def sort_entity(x):
    if not isinstance(x,str):
        return x
    temp = x.split(';')
    temp.sort()
    return ';'.join(temp)

def check_f1_score(df, negative_columns_true='negative',
                   key_entity_true='key_entity',
                   negative_columns_fit='negative_fit',
                   key_entity_fit='key_entity_fit'):
    ''' 计算训练集的f1score和accuracy
    '''
    df_i = df[df[negative_columns_true]==1].copy()
    df_j = df[df[negative_columns_true]==0].copy()
    
    df_i[key_entity_true] = df_i[key_entity_true].apply(sort_entity)
    df_i[key_entity_fit] = df_i[key_entity_fit].apply(sort_entity)
    
    TP = sum(df_i[key_entity_true]==df_i[key_entity_fit])
    TN = sum(df_j[negative_columns_fit]==0)
    P = TP/df[df[negative_columns_fit]==1].shape[0]
    R = TP/df[df[negative_columns_true]==1].shape[0]
    f1_score = (2*P*R)/(P+R)
    
    accuracy = (TP+TN)/df.shape[0]
    return f1_score, accuracy

def generate_key_entity(x, negative_entities,entity_column ='entity_x',
                        fit_column='negative_fit'):
    if not isinstance(x[entity_column], str) or x[fit_column]==0:
        return np.nan
    entity_list = x[entity_column].split(';')
    if len(entity_list) == 1:
        return entity_list[0]
    
    title = x['title'] if isinstance(x['title'], str) else ''
    title_split = x['title_split'] if isinstance(x['title_split'], str) else ''
    text = x['text'] if isinstance(x['text'], str) else ''
    text_split = x['txt_split'] if isinstance(x['txt_split'], str) else ''
    all_txt = text + text_split + title + title_split
    result_list = entity_list.copy()
    entity_list2 = entity_list.copy()
    for item in entity_list:
        if item not in all_txt and fuzz.partial_ratio(all_txt,item) < 50:
                entity_list2.remove(item)
    entity_list = entity_list2
        
    if len(entity_list) == 0:
        return ';'.join(result_list)
    elif len(entity_list) == 1:
        return ';'.join(entity_list)

    entity_list2 = entity_list.copy()
    for item_i in entity_list:
        for item_j in entity_list:
            if item_i != item_j and item_i in item_j and item_i in entity_list2:
                entity_list2.remove(item_i)
    entity_list = entity_list2
    
    if len(entity_list) == 1:
        return entity_list[0]

    result_list2 = entity_list.copy()
    for item in entity_list:
        if item not in negative_entities:
                result_list2.remove(item)
    result_list = result_list2
    if len(result_list) == 0:
        return ';'.join(entity_list)
    else:
        return ';'.join(result_list)

if __name__ == '__main__':
    predict_columns = ['glove','glove_word2vec','word2vec','bert']
    predict_columns_hastitle = ['glove_hastitle', 'glove_notitle',
                                'glove_word2vec_hastitle', 'glove_word2vec_notitle',
                                'word2vec_hastitle', 'word2vec_notitle',
                                'bert_hastitle', 'bert_notitle',]
#    predict_columns = ['glove_word2vec',]
#    predict_columns_hastitle = ['glove_word2vec_hastitle', 'glove_word2vec_notitle',]
    #%%1.load data
    data_train = pd.read_pickle(data_train_file)
    data_test = pd.read_pickle(data_test_file)
    
    data_train_hastitle = pd.read_pickle(data_train_file_hastitle)
    data_test_hastitle = pd.read_pickle(data_test_file_hastitle)
    
    get_data(data_train, data_test, data_train_hastitle, data_test_hastitle,)
    
    common_key = ['id']
    data_train_hastitle_new = pd.merge(data_train_hastitle, data_train[predict_columns+common_key],
                                       on = ['id',],how='left',suffixes=['_hastitle','_notitle'])
    data_test_hastitle_new = pd.merge(data_test_hastitle, data_test_hastitle[predict_columns+common_key],
                                      on = ['id',],how='left',suffixes=['_hastitle','_notitle'])
    
    
    data_train_new = data_train[pd.isnull(data_train['title'])].copy()
    data_test_new = data_test[pd.isnull(data_test['title'])].copy()
    
    #%%2.聚合各种模型
    clf_train = generate_predict_logistic(data_train_new, predict_columns)
    
    clf_train_hastitle = generate_predict_logistic(data_train_hastitle_new, predict_columns_hastitle)
    
    data_test_new.loc[:,'predict_logistic'] = clf_train.predict_proba(data_test_new[predict_columns].values)[:,1]
    data_test_hastitle_new.loc[:,'predict_logistic'] = clf_train_hastitle.predict_proba(data_test_hastitle_new[predict_columns_hastitle].values)[:,1]

    merge_columns = ['id', 'entity', 
                     'predict_logistic','title_split', 'txt_split']
    
#    data_test_new['predict_logistic'] = data_test_new.apply(generate_ensemble_lg, args=(predict_columns, 0.4,),axis=1)
#    data_train_hastitle_new['predict_logistic'] = data_train_hastitle_new.apply(generate_ensemble_lg, args=(predict_columns_hastitle, 0.4,),axis=1)
#    
#    data_test_new['predict_logistic'] = data_test_new.apply(generate_ensemble_lg, args=(predict_columns, 0.4,),axis=1)
#    data_test_hastitle_new['predict_logistic'] = data_test_hastitle_new.apply(generate_ensemble_lg, args=(predict_columns_hastitle, 0.4,),axis=1)
#    
    
    data_test_all = data_test_new[merge_columns].append(data_test_hastitle_new[merge_columns])
    data_train_all = data_train_new[merge_columns].append(data_train_hastitle_new[merge_columns])
    
    
    #%% 3.往旧数据中加入预测列
    common_key = ['id']
    # test file
    test_file_path = './data/Test_Data.csv'
    data_test_init = pd.read_csv(test_file_path)
    data_test_result = pd.merge(data_test_init, data_test_all[merge_columns],
                                on = common_key, how = 'left')
    # train file
    train_file_path = './data/Train_Data.csv'
    data_train_init = pd.read_csv(train_file_path)
    data_train_result = pd.merge(data_train_init, data_train_all[merge_columns],
                                 on = common_key, how = 'left',)
    data_train_result['negative_fit'] = (data_train_result.predict_logistic>0.5)*1
    print('accuracy',np.mean(data_train_result['negative_fit'] == data_train_result['negative']))
    
    
    ## get all negative 
    negative_entities = fetch_entites(data_train.query('negative==1'),['key_entity'],';')
    data_train_result['key_entity_fit'] = data_train_result.apply(generate_key_entity, 
                                     args=(negative_entities,'entity_x','negative_fit'),axis=1)
    check_f1_score(data_train_result)
    
    
    
    data_test_result['negative'] = (data_test_result.predict_logistic>0.5)*1
    data_test_result['key_entity'] = data_test_result.apply(generate_key_entity, 
                                     args=(negative_entities,'entity_x','negative'),axis=1)
    
    data_test_result.loc[pd.isnull(data_test_result.key_entity),'negative']=0
    
    data_test_result.to_csv('data_test_result.csv', encoding='utf_8_sig', index=False)
    data_test_result[['id','negative','key_entity']].to_csv('data_test_submit.csv', encoding='utf_8_sig', index=False)
    
    
    data_train_result.to_csv('data_train_result.csv',encoding='utf_8_sig', index=False)
#    sum(data_train_result['key_entity_fit2']==data_train_result['key_entity_init'])/data_train_result.shape[0]
#    data_train_result[['id','negative','key_entity']].to_csv('data_train_submit.csv',encoding='utf_8_sig', index=False)
    
    