#encoding:utf-8
'''import collections
c=collections.Counter()
print c
sentences=[["hello","nihao","xixi","hello"],['hi','hi','lin']]
for sent in sentences:
    c.update(sent)
print c
print c.most_common()

colors = ['red', 'blue', 'red', 'green', 'blue', 'blue']
c = collections.Counter(colors)
print (dict(c))'''

import cPickle as pickle
import numpy as np
f = open('/Users/lin/PycharmProjects/lin/nlp/data/vocab.zh.pkl')
info = pickle.load(f)
print "vocab.zn.pkl:",info  # show file
print len(info)

import numpy as np
B=np.load("nnlm_word_embeddings.zh.npy")
B=B.tolist()
#print "nnlm_word_embeddings.zh.npy:",B
print type(B)
#print B.shape

'''en=np.load("nnlm_word_embeddings.en.npy")
print en
print type(en)
print en.shape'''

'''print [2] * 5'''

import os
import pandas as pd

x_name = []
x_name.append("word")
x_name.append("word_vector")
output=pd.DataFrame(columns=x_name)
output.to_csv("./wordEmbedding.txt",index=False)  #把一个DataFrame写入csv文件


f = open("./wordEmbedding.txt", 'a')
for i in range(len(info)):
    word=[info[i]]
    #print type(word)
    word_vector=B[i]
    #print type(word_vector)
    write=word+word_vector
    f.write(str(word).replace('[', '').replace(']', '')+ '\t' + str(word_vector).replace('[', '').replace(']', '').replace(',',' ') + '\n')

'''word=[info[2]]
#print type(word)
word_vector=B[2]
#print type(word_vector)
write=word+word_vector
print str(write).strip('[]')
word=[info[3]]
#print type(word)
word_vector=B[3]
#print type(word_vector)
write=word+word_vector
print str(write).strip('[]')'''

'''f = open("./wordEmbedding.csv", 'a')
for i in range(len(info)):
    word=info[i]
    print str(word)
    #print type(word)
    #print word
    word_vector=B[i]
    #print type(word_vector)
    word_vector = str(word_vector).replace('[', '').replace(']', '')  # 去除[]
    print str(word_vector)

    write=word+word_vector
    if(i==5):
        break
    f.write(str(word) + str(word_vector) + '\n')'''

