#encoding:utf-8
import cPickle as pickle
import numpy as np
f = open('/Users/lin/PycharmProjects/lin/nlp/data/vocab.en.pkl')
info = pickle.load(f)

import numpy as np
B=np.load("nnlm_word_embeddings.en.npy")
B=B.tolist()

import os
import pandas as pd

'''x_name = []
x_name.append("word")
x_name.append("word_vector")
output=pd.DataFrame(columns=x_name)
output.to_csv("./en_wordEmbedding.txt",index=False)  #把一个DataFrame写入csv文件'''


f = open("./en_wordEmbedding.txt", 'a')
for i in range(len(info)):
    word=info[i]
    #print type(word)
    word_vector=B[i]
    #print type(word_vector)
    #write=word+word_vector
    f.write(word+ '\t' + str(word_vector).replace('[', '').replace(']', '').replace(',',' ') + '\n')
