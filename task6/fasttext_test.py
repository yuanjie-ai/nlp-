#!/usr/bin/env python
# coding: utf-8

# In[1]:


import fastText


# In[2]:


help(fastText.train_supervised)


# In[3]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import jieba


# In[4]:


def get_stopwords(path):
#     stopwords = pd.read_csv(path,sep='\n',engine='python',names=['stopword'],encoding='utf-8')
#     stopwords = stopwords.values
    
    with open(path,encoding='utf-8') as f:
        stopwords = f.read().strip().split('\n')
    
    return stopwords


# In[5]:


label_dict = {'体育':0, '娱乐':1, '家居':2, '房产':3, '教育':4, '时尚':5, '时政':6, '游戏':7, '科技':8, '财经':9}


# In[24]:


def preprocess_data(data_path,save_path,encoder,stopwords):
    data = pd.read_csv(data_path,sep='\t',engine='python',names=['label','content'],encoding='utf-8')
    # 标签通过字典映射后再取值
    labels = data['label'].map(label_dict).values
    
    contents = data['content'].values
    preprocess_contents = []
    text = []
    for label,content in zip(labels,contents):
        content = jieba.lcut(content)
        content = filter(lambda x:len(x)>1, content)
        content = filter(lambda x:x not in stopwords, content)
        preprocess_contents.append(content)
        text.append('__label__'+ str(label) + ' ' + ' '.join(content)+'\n')
    with open(save_path,'w',encoding='utf-8') as f:
        for i in text:
            f.write(i)
    

    return contents,labels



# stopwords = get_stopwords('./stopwords.txt')
#
# # 暂时不用encoder了
# encoder = LabelEncoder()
# contents,labels = preprocess_data('../dataset/cnews/cnews.train.txt','./after_reprocess_train.txt',encoder,stopwords)
# print(contents[0],type(contents))
#

model = fastText.FastText.train_supervised('./after_reprocess_train.txt',lr=0.05, dim=300, epoch=20,label='__label__')
res = model.test('./after_reprocess_train.txt')

print(res)


model.save_model('./fasttest.model')


test = ['姚明','未来','次节', '出战', '成疑', '火箭', '高层', '改变', '用姚', '战略', '副总裁', '球员' ,'凯尔', '洛瑞' ,'伤病','防守' ,'篮板']



test = ['陈云林', '陆委会', '主委', '赖幸媛', '见面', '人民网', '11', '快讯', '协会会长', '陈云林', '今天下午', '前往', '台北', '晶华', '酒店', '正在', '陆委会', '主委', '赖幸媛', '见面', '据悉', '对于', '外界', '关心', '双方', '称谓', '问题', '陈云林', '幸媛', '天天', '电视', '看到', '高兴', '见到']



label = model.predict(test,k=1)
print(label)


print(model.get_word_vector('录取'))





