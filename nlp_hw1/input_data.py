#encoding:utf-8

#import sys
#reload(sys)
#sys.setdefaultencoding('utf8')

import os
import codecs
import collections
from six.moves import cPickle
import numpy as np
import re
import itertools


class TextLoader():
    def __init__(self, data_dir, batch_size, seq_length, mini_frq=3):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.mini_frq = mini_frq

        input_file = os.path.join(data_dir, "input.zh.txt")
        vocab_file = os.path.join(data_dir, "vocab.zh.pkl")
        '''input_file = os.path.join(data_dir, "input.en.txt")
        vocab_file = os.path.join(data_dir, "vocab.en.pkl")'''

        self.preprocess(input_file, vocab_file)
        self.create_batches()
        self.reset_batch_pointer()

    def build_vocab(self, sentences):
        word_counts = collections.Counter()  #统计词频，先创建一个空的计数器
        if not isinstance(sentences, list): #判断一个对象是否是一个已知的类型，类似 type()
            sentences = [sentences]
        for sent in sentences:
            word_counts.update(sent)  #更新词频
        vocabulary_inv = ['<START>', '<UNK>', '<END>'] + \
                         [x[0] for x in word_counts.most_common() if x[1] >= self.mini_frq]#most_common() 类似topn函数，返回前n个出现次数最多的,没有参数时返回全部
        vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
        return [vocabulary, vocabulary_inv]

    def preprocess(self, input_file, vocab_file):
        with codecs.open(input_file, 'r', 'utf-8') as f:
            lines = f.readlines()
            #print "readlines:",lines
            if lines[0][:1] == codecs.BOM_UTF8:
                lines[0] = lines[0][1:]
            lines = [line.strip().split() for line in lines] #去掉首尾空符，并按照空格分割
            #print "lines:",lines


        self.vocab, self.words = self.build_vocab(lines)  #用输入文件生成词频表
       # print "vocab:",self.vocab    #字典 词汇：词频
        #print "type_vocab",type(self.vocab)
       # print "words:",self.words  #词汇
        self.vocab_size = len(self.words)
       # print 'word num: ', self.vocab_size

        with open(vocab_file, 'wb') as f:
            cPickle.dump(self.words, f) #存储数据集 将words存储到"vocab.zh.pkl"中
#self.vocab.get(w, 1)返回w对应的值，若不存在返回1
        raw_data = [[0] * self.seq_length +
            [self.vocab.get(w, 1) for w in line] +
            [2] * self.seq_length for line in lines]

        '''for i in range(10):
            print "rawdata:",raw_data[i]
            print "raw_data[%d]_len:" % i,len(raw_data[i])
        print "len_rawdata",len(raw_data)'''
        self.raw_data = raw_data  #根据生成的词频表将输入文件中的单词映射到数值（即词频）
        #列表套列表，大列表共有句子总数个元素，每个小元素由seq_length个0+一句话中每个词的index+seq_length个2构成

    def create_batches(self):
        xdata, ydata = list(), list()
        for row in self.raw_data:
            for ind in range(self.seq_length, len(row)):
                xdata.append(row[ind-self.seq_length:ind]) # xdata为raw_data的第0-seq_lenth个
                ydata.append([row[ind]])  # ydata即为xdata后一个单词
        self.num_batches = int(len(xdata) / self.batch_size)
        if self.num_batches == 0:
            assert False, "Not enough data. Make seq_length and batch_size small."

        xdata = np.array(xdata[:self.num_batches * self.batch_size])
        ydata = np.array(ydata[:self.num_batches * self.batch_size])

        self.x_batches = np.split(xdata, self.num_batches, 0)
        self.y_batches = np.split(ydata, self.num_batches, 0)

    def next_batch(self):
        x, y = self.x_batches[self.pointer], self.y_batches[self.pointer]
        self.pointer += 1
        return x, y

    def reset_batch_pointer(self):
        self.pointer = 0
