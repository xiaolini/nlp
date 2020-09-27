#encoding:utf-8
import cPickle as pickle
import numpy as np
f = open('/Users/lin/PycharmProjects/lin/nlp/data/vocab.zh.pkl')
info = pickle.load(f)
'''print "vocab.zn.pkl:",info  # show file
print len(info)
print type(info[0])
print info[0]
print type(info[3])
print info[3]
print info[4]
print type(info[4])
print info[4].decode('unicode_escape')
'''
import numpy as np
B=np.load("nnlm_word_embeddings.zh.npy")
B=B.tolist()
#print "nnlm_word_embeddings.zh.npy:",B
print type(B)
#print B.shape


import os
import pandas as pd

'''x_name = []
x_name.append("word")
x_name.append("word_vector")
output=pd.DataFrame(columns=x_name)
output.to_csv("./zh_wordEmbedding.txt",index=False)  #把一个DataFrame写入csv文件'''


f = open("./zh_wordEmbedding.txt", 'a')
for i in range(len(info)):
    word=info[i]
    #print type(word)
    word_vector=B[i]
    #print type(word_vector)
    #write=word+word_vector
    if i==0 or i==1 or i==2:
        f.write(word + '\t' + str(word_vector).replace('[', '').replace(']', '').replace(',', ' ') + '\n')
    else:
        f.write(word.encode('utf-8') + '\t' + str(word_vector).replace('[', '').replace(']', '').replace(',', ' ') + '\n')

