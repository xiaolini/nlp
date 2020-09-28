#encoding:utf-8
import sys, pickle, os, random
import numpy as np
import io

## tags, BIO
tag2label = {"O": 0,
             "B-PER": 1, "I-PER": 2,
             "B-LOC": 3, "I-LOC": 4,
             "B-ORG": 5, "I-ORG": 6
             }


def read_corpus(corpus_path):
    """
    read corpus and return the list of samples
    :param corpus_path:
    :return: data
    """
    data = []
    with io.open(corpus_path, encoding='utf-8') as fr:
        lines = fr.readlines()
    sent_, tag_ = [], []
    for line in lines:
        if line != '\n':
            [char, label] = line.strip().split()
            sent_.append(char)
            tag_.append(label)
        else:
            data.append((sent_, tag_))
            sent_, tag_ = [], []

    return data


def vocab_build(vocab_path, corpus_path, min_count):
    """
    :param vocab_path:
    :param corpus_path:
    :param min_count:
    :return:
    """
    data = read_corpus(corpus_path)
    word2id = {}  #创建一个字典
    for sent_, tag_ in data:
        for word in sent_:
            if word.isdigit(): #数字
                word = '<NUM>'
            elif ('\u0041' <= word <='\u005a') or ('\u0061' <= word <='\u007a'): #英文字母
                word = '<ENG>'
            if word not in word2id:
                word2id[word] = [len(word2id)+1, 1]  #如果这个词没出现过，为这个词创建id，id为当前word2id中标号加1，出现次数记1
            else:
                word2id[word][1] +=1    #统计单词出现频数
    low_freq_words = []
    for word, [word_id, word_freq] in word2id.items():  #统计低频词
        if word_freq < min_count and word != '<NUM>' and word != '<ENG>':
            low_freq_words.append(word)
    for word in low_freq_words: #在word2id中删除低频词
        del word2id[word]

    new_id = 1
    for word in word2id.keys():#删除低频词后 对词重新分配id
        word2id[word] = new_id
        new_id += 1
    word2id['<UNK>'] = new_id
    word2id['<PAD>'] = 0

    print(len(word2id))
    with open(vocab_path, 'wb') as fw:
        pickle.dump(word2id, fw)


def sentence2id(sent, word2id):
    """
    :param sent:
    :param word2id:
    :return:
    """
    sentence_id = []
    for word in sent:
        if word.isdigit():
            word = '<NUM>'
        elif ('\u0041' <= word <= '\u005a') or ('\u0061' <= word <= '\u007a'):
            word = '<ENG>'
        if word not in word2id:
            word = '<UNK>'
        sentence_id.append(word2id[word])
    return sentence_id


def read_dictionary(vocab_path):
    """
    :param vocab_path:
    :return:
    """
    vocab_path = os.path.join(vocab_path) #路径连接函数
    with open(vocab_path, 'rb') as fr:
        word2id = pickle.load(fr)
    print('vocab_size:', len(word2id))
    return word2id


def random_embedding(vocab, embedding_dim):
    """
    :param vocab:
    :param embedding_dim:
    :return:
    """
    embedding_mat = np.random.uniform(-0.25, 0.25, (len(vocab), embedding_dim))
    embedding_mat = np.float32(embedding_mat)
    return embedding_mat


def pad_sequences(sequences, pad_mark=0):
    """
    :param sequences:
    :param pad_mark:
    :return:
    """
    max_len = max(map(lambda x : len(x), sequences)) #求sequence中每一个元素的长度，再取最大值
    seq_list, seq_len_list = [], []
    for seq in sequences:
        seq = list(seq)
        seq_ = seq[:max_len] + [pad_mark] * max(max_len - len(seq), 0)  #以最长序列为基准，长度小的序列补零
        seq_list.append(seq_)
        seq_len_list.append(min(len(seq), max_len))#记录原始真实长度
    return seq_list, seq_len_list

#生成batch
''' seqs的形状为二维矩阵，形状为[[33,12,17,88,50....]...第一句话
                                [52,19,14,48,66....]...第二句话
                                                    ] 
   labels的形状为二维矩阵，形状为[[0, 0, 3, 4]....第一句话
                                 [0, 0, 3, 4]...第二句话
                                             ]
'''
def batch_yield(data, batch_size, vocab, tag2label, shuffle=False):
    """
    :param data:
    :param batch_size:
    :param vocab:
    :param tag2label:
    :param shuffle:
    :return:
    """
    if shuffle:
        random.shuffle(data)
    # sent_是一个列表(文本序列)，tag也是一个列表(文本序列对应的标签)
    # 使用sentence2id将文本序列映射为数值序列，为自己定义的一个文本处理函数
    seqs, labels = [], []
    for (sent_, tag_) in data:
        sent_ = sentence2id(sent_, vocab)
        label_ = [tag2label[tag] for tag in tag_]

        if len(seqs) == batch_size:
            yield seqs, labels
            seqs, labels = [], []

        seqs.append(sent_)
        labels.append(label_)

    if len(seqs) != 0:
        yield seqs, labels
#yield在python 里就是一个生成器。当你使用一个yield的时候，对应的函数就是一个生成器了。
#yield就是 return 返回一个值，并且记住这个返回的位置，下次迭代就从这个位置后(下一行)开始。带有yield的函数不仅仅只用于for循环中，而且可用于某个函数的参数，只要这个函数的参数允许迭代参数。
# batches=batch_yield(train_data, batch_size, vocab, tag2label, shuffle=False）的类型是一个generator
#在Python中，一边循环，一边计算的机制，称为生成器。每当生成器被调用的时候，它会返回一个值给调用者。

#生成word2id.pkl词汇表编号文件
'''data = read_corpus('./data_path/train_data.txt')
print type(data)

vocab_build('./data_path/word2id.pkl','./data_path/train_data.txt',1)'''