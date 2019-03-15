import tensorflow as tf
import pandas as pd
import numpy as np
from gensim.models.word2vec import Word2Vec
import jieba
import re


label_dict = {'体育':0, '娱乐':1, '家居':2, '房产':3, '教育':4,
              '时尚':5, '时政':6, '游戏':7, '科技':8, '财经':9}

def get_stopwords(path):
    with open(path,encoding='utf-8') as f:
        stopwords = f.read().split('\n')
    return stopwords

def sentence_preprocess(x,stopwords):
    instopwords = lambda word: True if word in stopwords else False
    hasillegal = lambda word:True if re.match(r'[\W]',word) else False

    data = []
    for sentence in x:
        words = []
        for word in jieba.lcut(sentence):
            if instopwords(word) or word.isdigit() or hasillegal(word) or len(word)<2:
                continue
            words.append(word)
        data.append(words)

    return data

def cnews_data_preprocess(path,stopwords_path,save_path):
    data = pd.read_csv(path,sep='\t',names=['label','content'],engine='python',encoding='utf-8')
    y = data['label'].map(label_dict).values
    x = data['content'].values
    stopwords = get_stopwords(stopwords_path)
    x = sentence_preprocess(x,stopwords)

    with open(save_path,'w',encoding='utf-8') as f:
        for label,sents in zip(data['label'].values,x):
            f.write(label+','+ ' '.join(sents)+'\n')
    return x,y

def read_after_prep_data(path):
    data = pd.read_csv(path, sep=',', names=['label', 'content'], engine='python', encoding='utf-8')
    y = data['label'].map(label_dict).values
    x = []
    for i in data['content'].values:
        x.append(i.split(' '))
    x = np.array(x)
    return x,y

def data2wordvec(x,model,max_seqlen=50):
    vectors = []
    for sentence in x:
        count = 0
        vec = []
        for word in sentence:
            try:
                if count >= max_seqlen:
                    break
                vec.append(model.wv[word].tolist())
                count += 1
            except KeyError:
                continue

        for _ in range(max_seqlen-count):
            padding = [0]*model.wv.vector_size
            vec.append(padding)
        vectors.append(vec)
    return vectors

def get_w2v_model(path):
    model = Word2Vec.load(path)
    return model

def train_w2v(path,x):
    w2v = Word2Vec(size=300)
    w2v.build_vocab(x)
    w2v.train(x,total_examples=w2v.corpus_count,epochs=w2v.epochs)
    w2v.save(path)
    return w2v