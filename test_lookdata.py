#encoding:utf-8
import sys
reload(sys)
sys.setdefaultencoding('utf8')

from input_data import *

import numpy as np
import tensorflow as tf
import argparse
import time
import math
from tensorflow.python.platform import gfile



parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='/Users/lin/PycharmProjects/lin/nlp/data/',
                   help='data directory containing input.txt')
parser.add_argument('--batch_size', type=int, default=120,
                   help='minibatch size')
parser.add_argument('--win_size', type=int, default=5,
                   help='context sequence length')
parser.add_argument('--hidden_num', type=int, default=64,
                   help='number of hidden layers')
parser.add_argument('--word_dim', type=int, default=50,
                   help='number of word embedding')
parser.add_argument('--num_epochs', type=int, default=10,
                   help='number of epochs')
parser.add_argument('--grad_clip', type=float, default=10.,
                   help='clip gradients at this value')

args = parser.parse_args() #参数集合

#准备训练数据
data_loader = TextLoader(args.data_dir, args.batch_size, args.win_size)
args.vocab_size = data_loader.vocab_size
weight_h = tf.Variable(tf.truncated_normal([args.win_size * args.word_dim + 1, args.hidden_num],
                            stddev=1.0 / math.sqrt(args.hidden_num)))
print weight_h
data_loader.reset_batch_pointer()
for b in range(data_loader.num_batches):
    start = time.time()
    x, y = data_loader.next_batch()
    #print("x:",x)
    #print("y:",y)
    #print x.shape,y.shape
    #break