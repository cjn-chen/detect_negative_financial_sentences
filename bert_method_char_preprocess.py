from keras_bert import Tokenizer
import pandas as pd
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle
import codecs
from data_preprocess import fetch_entites

def get_token_dict(dict_path):
    ''' 构造token字典
    Args:
        dict_path:token字典文件，每行为一个对应的可以
    '''
    token_dict = {}
    with codecs.open(dict_path, 'r', 'utf8') as reader:
        for line in reader:
            token = line.strip()
            token_dict[token] = len(token_dict)
    return token_dict

def get_token(x, tokenizer):
    ''' 因为此处为单句输入，segment全部都是0,不必返回
    '''
    tokens, segment = tokenizer.encode(x)
    return tokens

def get_tokens(data, token_field, max_len_percent, tokenizer):
    ''' 输入df以及需要token的字段（每个元素为str），返回对应的tokenize后的矩阵
    Args:
        data:数据通过pd载入后的DataFrame
        token_field:需要token的字段
        max_len_percent:需要截取的该字段的最大长度，下分位数，如95%表示截取长度的95%分位数
    Returns:
        返回对应的tokenize后的矩阵
    '''
    find_maxlen = np.array([len(i.replace(' ','')) for i in data[token_field]])
    # 取所有text的95%长度为最大长度
    maxlen = int(np.percentile(find_maxlen, max_len_percent))
    token_first = data[token_field].apply(get_token, args=(tokenizer,))
    tokens_mat = pad_sequences(token_first, max(maxlen,2), padding='post')
    return tokens_mat

def get_tokens_entity(x, tokenizer, maxlen):
    ''' 输入df以及需要token的字段（每个元素为str），返回对应的tokenize后的矩阵
    Args:
        data:数据通过pd载入后的DataFrame
        token_field:需要token的字段
        max_len_percent:需要截取的该字段的最大长度，下分位数，如95%表示截取长度的95%分位数
    Returns:
        返回对应的tokenize后的矩阵
    '''
    if not isinstance(x, str):
        return [101, 102]
    entity_list = x.split(';')
    token_entity_list = []
    for i in entity_list:
        token_first = get_token(i, tokenizer)
        token_entity_list.append(token_first)
    tokens_mat = pad_sequences(token_entity_list, max(maxlen,2), padding='post')
    return tokens_mat
    
def generate_training_data(data_train_file, output_file, tokenizer):
    ''' 生成tokenize后的数据
    Args:
        data_train_file:训练集文件
            output_file:输出的tokenize后的文件
            tokenizer:tokenizer用于tokenize的类
    Returns:
        shape_dic:由generate_training_data生成的txt，title, 的shape
    '''
    data_train = pd.read_pickle(data_train_file)
    
    entities = fetch_entites(data_train, ['entity','key_entity'], ';')
    max_entity_len = 0
    for i in entities:
        max_entity_len = max(max_entity_len, len(i))
#    print('tokenize training set...')
    X_train_txt = get_tokens(data_train, 'text', 95, tokenizer)
    data_train['title2'] = data_train.title.fillna('')
    X_train_title = get_tokens(data_train, 'title2', 95, tokenizer)
    
    entity_tokens_mat = data_train.entity.apply(get_tokens_entity, 
                                                args=(tokenizer,max_entity_len))
    key_entity_tokens_mat = data_train.key_entity.apply(get_tokens_entity, 
                                                    args=(tokenizer,max_entity_len))
    y_train = data_train.negative.values
#    print('save training set file...')
    ## save data
    train_data = dict(zip(['txt', 'title', 'entity','key_entity',
                           'y_train'], 
                          [X_train_txt, X_train_title, entity_tokens_mat,
                           key_entity_tokens_mat,
                           y_train]))
    with open(output_file, 'wb') as f:
        pickle.dump(train_data, f)
    shape_dic = {'txt_shape':X_train_txt.shape[1], 
                 'title_shape':X_train_title.shape[1],
                 'entity_shape':max_entity_len}
    return shape_dic

def generate_test_data(data_test_file, output_file, tokenizer, shape_dic):
    ''' 生成tokenize后的数据
    Args:
        data_test_file:test set 文件
            output_file:输出的tokenize后的文件
            shape_dic:由generate_training_data生成的txt，title的shape
    '''
    data_test = pd.read_pickle(data_test_file)
    print('tokenize test set...')
    data_test['text2'] = data_test.text.fillna('')
    X_test_txt = get_tokens(data_test, 'text2', 95, tokenizer)
    data_test['title2'] = data_test.title.fillna('')
    X_test_title = get_tokens(data_test, 'title2', 95, tokenizer)
    
    max_entity_len = shape_dic['entity_shape']
    
    entity_tokens_mat = data_test.entity.apply(get_tokens_entity, 
                                               args=(tokenizer,max_entity_len))
    
    # 保证test set的padding长度 和train set一致
    if shape_dic['txt_shape'] > X_test_txt.shape[1]:
        X_test_txt = pad_sequences(X_test_txt, shape_dic['txt_shape'], padding='post')
    else:
        X_test_txt = X_test_txt[:,:shape_dic['txt_shape']]
        
    if shape_dic['title_shape'] > X_test_title.shape[1]:
        X_test_title = pad_sequences(X_test_title, shape_dic['title_shape'], padding='post')
    else:
        X_test_title = X_test_title[:,:shape_dic['title_shape']]
    
    print('save test set file...')
    ## save data
    test_data = dict(zip(['txt', 'title', 'entity'], 
                          [X_test_txt, X_test_title, entity_tokens_mat]))
    with open(output_file, 'wb') as f:
        pickle.dump(test_data, f)


if __name__ == '__main__':
    # 1.获取token字典
    print('load token dict from file...')
    dict_path = './chinese_L-12_H-768_A-12/vocab.txt'
    token_dict = get_token_dict(dict_path)
    tokenizer = Tokenizer(token_dict)
    
    ## 1.2进行tokenizer
    ### no title
    print('produce training set...')
    data_train_file = 'Train_Data.pkl'
    output_file = 'train_data_model_bert_char.pkl'
    shape_dic = generate_training_data(data_train_file, output_file, tokenizer)
    
    print('produce test set...')
    data_test_file = 'Test_Data.pkl'
    output_file = 'test_data_model_bert_char.pkl'
    generate_test_data(data_test_file, output_file, tokenizer, shape_dic)

    ### has title
    print('produce training set (has tille)...')
    data_train_file = 'Train_Data_hastitle.pkl'
    output_file = 'train_data_model_hastitle_bert_char.pkl'
    shape_dic = generate_training_data(data_train_file, output_file, tokenizer)
    
    print('produce test set (has tille)...')
    data_test_file = 'Test_Data_hastitle.pkl'
    output_file = 'test_data_model_hastitle_bert_char.pkl'
    generate_test_data(data_test_file, output_file, tokenizer, shape_dic)
