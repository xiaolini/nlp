#encoding:utf-8
'''sequences=[[1,2,3],[1,2],[1],[12,3,1]]
max_len = max(map(lambda x : len(x), sequences)) #求sequence中每一个元素的长度，再取最大值
seq_list, seq_len_list = [], []
for seq in sequences:
    seq = list(seq)
    seq_ = seq[:max_len] + [0] * max(max_len - len(seq), 0)  #长度小于maxlen，补零
    seq_list.append(seq_)
    seq_len_list.append(min(len(seq), max_len))
print seq_list,seq_len_list'''
'''from data import read_dictionary
import os
word2id = read_dictionary(os.path.join('.', 'data_path', 'word2id.pkl'))
print word2id'''

import tensorflow as tf

a = tf.sequence_mask([13])
b = tf.sequence_mask([[1, 2], [7, 8]])

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(a))
    print(sess.run(b))
