import re
import pandas as pd
import numpy as np
#from gensim.models import word2vec
import jieba
# from fuzzywuzzy import fuzz

#  去掉中英文状态下的逗号、句号
stop_words_clear_num_english = ['上','下','.','日','月','年',
                                '元','万','亿',
                                '万元','亿元','万港元','亿港元',
                                '万美元','亿美元','千万',
                                '港元','美元', 
                                '十亿元','百亿元','十亿港元','百亿港元',
                                '十亿美元','百亿美元',
                                '十万元','百万元','十万港元','百万港元',
                                '十万美元','百万美元',
                                '①','②','③','④','⑤','⑥',]
chinese_num = ['一','二','三','四','五','六','七','八',
               '九','十','零','万','千','亿']

def clearSen(comment):
    ''' 输入句子,输出清理了特殊字符后的句子
    '''
    if isinstance(comment, str):
        txt = re.sub(r'[\?]', '', comment) # 去除识别错误的问号?(英文字符)
        txt = re.sub(r'\[超话\]',' ',txt)  # 将[超话]变为一个空格
        txt = re.sub(r'&quot|&gt|&nbsp|…|<p>|<//p>|<\/a>|<a>|<\/p>|<strong>|'
                     +r'<\/strong>|<\/articlead>|<articlead>|< articlead>|IMG:\d|IMG \d|'
                     +r'<header>|<\/header>|<div>|<\/div>|<span>|<\/span>|'
                     +r'<br>|<\/br>|<article>|<\/article>|<P>|<\/P>',' ',txt)  # 将引号变为一个空格
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
        txt = re.sub(r'[\*『』◆▲+▼「」@#《》〖〗【】，,。？！!“"”；;·\-：:\[\]\/～~、丶 ()（）｜\|_丨→\{\}｛｝]',' ',txt)
        txt = re.sub(r'[―—]',' ',txt)  # 将特殊字符变为一个空格
        txt = re.sub(r'\s+',' ',txt)  # 将多个空格变为1个
        return txt.strip()
    else:
        return comment
    
def clear_stop_word_train_data(x, stop_words, field_name):
    ''' 删除停用词
    Args:
        x:清除停词的字段
        stop_words:停词表
    '''
    entity = x['entity'] if isinstance(x['entity'],str) else ''
    key_entity = x['key_entity'] if isinstance(x['key_entity'],str) else ''
    filter_field = x[field_name]
    for i in stop_words:
        if isinstance(filter_field, str):
            if i in filter_field and i not in entity and i not in key_entity:
                filter_field = filter_field.replace(i,'')
    return filter_field

def clear_stop_word_test_data(x, stop_words, field_name):
    ''' 删除停用词
    Args:
        x:清除停词的字段
        stop_words:停词表
    '''
    entity = x['entity'] if isinstance(x['entity'],str) else ''
    filter_field = x[field_name]
    for i in stop_words:
        if isinstance(filter_field, str):
            if i in filter_field and i not in entity:
                filter_field = filter_field.replace(i,'')
    return filter_field

def clear_stop_num_english(x, field_name):
    ''' 删除停用词
    Args:
        x:清除停词的字段
        stop_words:停词表
    '''
    
    entity = x['entity'] if isinstance(x['entity'],str) else ''
    filter_field = x[field_name]
    if not isinstance(filter_field, str):
        return filter_field
    filter_field = filter_field.replace('\n', '')
    for i in filter_field.split():
        ## 剔除所有非中英文，非公司名符号
        symbol_txt = re.search('[^\w\u4e00-\u9fff]+',i)
        if symbol_txt:
            if len(symbol_txt.group()) == len(i) and i not in entity:
                filter_field = filter_field.replace(i, '')
                continue
        ## 剔除全英文非公司名部分
        enlish_txt = re.search(r'([A-Z]|[a-z]|\d)*', i)
        if enlish_txt:
            if len(enlish_txt.group()) == len(i) and i not in entity:
                filter_field = filter_field.replace(i, '')
                continue
        ## 剔除数字非公司名部分
        new_i = i.replace('%','')         
        try:
            new_i = float(new_i)
            if isinstance(new_i, float) and i not in entity:
                filter_field = filter_field.replace(i, '')
                continue
        except:
            pass
        ## 剔除中文数字非公司名部分
        is_all_chinese = True
        for new_i_item in i:
            if new_i_item not in chinese_num:
                is_all_chinese = False
        if is_all_chinese:
            filter_field = filter_field.replace(i, '')
            continue
        ## 剔除指定停词
        if i in stop_words_clear_num_english and i not in entity:
            filter_field = filter_field.replace(i, '')
    filter_field = re.sub(r'\s+',' ',filter_field) # 将多个空格变为1个
    filter_field = filter_field.strip()
    return filter_field

def fetch_entites(df, df_name, sep):
    ''' 输入关于entity的某一列，输出entity的set
    args:
        df:需要分析的dataframe
        df_name:需要提取entity的列名，列表
        sep:分隔符
    '''
    all_entity = set()
    for df_name_i in df_name:
        for entity_txt in df[df_name_i]:
            if isinstance(entity_txt, str):
                entities = set(entity_txt.split(sep))
                if len(entities) > 0:
                    for entity in entities:
                        if len(entity) > 0:
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
    
def output_file(set_to_write, output_file_path):
    ''' 
    '''
    file = open(output_file_path, 'w', encoding = 'UTF-8')
    for item in set_to_write:
        if isinstance(item, str):
            if len(item) >0:
                file.write(item+'\n')
    file.close()
    
if __name__ == '__main__':
    #%%1.数据清洗
    train_file_path = './data/Train_Data.csv'
    data_train = pd.read_csv(train_file_path)
    data_train['title'] = data_train['title'].apply(clearSen)
    data_train['text'] = data_train['text'].apply(clearSen)
    
    test_file_path = './data/Test_Data.csv'
    data_test = pd.read_csv(test_file_path)
    data_test['title'] = data_test['title'].apply(clearSen)
    data_test['text'] = data_test['text'].apply(clearSen)
    ## 1.1 停词清洗
    with open('./data/stop_word.txt','r',encoding='utf-8') as f:
        stopwords = f.readlines()
        stopwords = [i.replace('\n','') for i in stopwords]
    stopwords = set(stopwords)
    data_train['title'] = data_train.apply(clear_stop_word_train_data, args=(stopwords,'title'),axis=1)
    data_train['text'] = data_train.apply(clear_stop_word_train_data, args=(stopwords,'text'),axis=1)
    
    data_test['title'] = data_test.apply(clear_stop_word_test_data, args=(stopwords,'title'), axis=1)
    data_test['text'] = data_test.apply(clear_stop_word_test_data, args=(stopwords,'text'), axis=1)
    #%% 2提取entities
    ## 用于jieba进行分词使用
    entities = fetch_entites(data_train, ['entity','key_entity'], ';')
    if '' in entities:
        entities.remove('')
    with open('financial_entity.txt', 'w', encoding = 'UTF-8') as f:
        for entity in entities:
            if len(entity) > 0:
                f.write(entity+' 999\n')
   ## 2.1载入分词，调整分词
    jieba.load_userdict('financial_entity.txt')
    for j in range(2):
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
    ## 2.2将分割后的结果保存在训练集中
    data_train['title_split'] = data_train['title'].apply(seperate_txt)
    data_train['txt_split'] = data_train['text'].apply(seperate_txt)
    ## 剔除数字和非公司名英文
    data_train['title_split'] = data_train.apply(clear_stop_num_english, args=('title_split',),axis=1)
    data_train['txt_split'] = data_train.apply(clear_stop_num_english, args=('txt_split',),axis=1)
    
    data_train.to_pickle('./Train_Data.pkl')
    data_train[~pd.isnull(data_train.title)].to_pickle('./Train_Data_hastitle.pkl')

    #%%3 加入测试集的entity
    entities = fetch_entites(data_test, ['entity',], ';')
    if '' in entities:
        entities.remove('')
    with open('financial_entity_test.txt', 'w', encoding = 'UTF-8') as f:
        for entity in entities:
            if len(entity) > 0:
                f.write(entity+' 999\n')
    jieba.load_userdict('financial_entity_test.txt')
    ## 3.1将分割后的结果保存在测试集中
    data_test['title_split'] = data_test['title'].apply(seperate_txt)
    data_test['txt_split'] = data_test['text'].apply(seperate_txt)
    ## 剔除数字和非公司名英文
    data_test['title_split'] = data_test.apply(clear_stop_num_english, args=('title_split',),axis=1)
    data_test['txt_split'] = data_test.apply(clear_stop_num_english, args=('txt_split',),axis=1)

    data_test.to_pickle('./Test_Data.pkl')
    data_test[~pd.isnull(data_test.title)].to_pickle('./Test_Data_hastitle.pkl')
    #%%4.输出分割的句子
#    data_train = pd.read_pickle('./Train_Data.pkl')
#    data_test = pd.read_pickle('./Test_Data.pkl')
    all_word_seg=set(data_train['title_split'].unique())
    all_word_seg = all_word_seg.union(set(data_train['txt_split'].unique()))
    if np.nan in all_word_seg:
        all_word_seg.remove(np.nan)
    output_file(all_word_seg, './all_word_seg_train.txt')
    
    all_word_seg = all_word_seg.union(set(data_test['title_split'].unique()))
    all_word_seg = all_word_seg.union(set(data_test['txt_split'].unique()))
    if np.nan in all_word_seg:
        all_word_seg.remove(np.nan)
    output_file(all_word_seg, './all_word_seg.txt')
