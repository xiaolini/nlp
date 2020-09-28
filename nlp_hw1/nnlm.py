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

def main():
    parser = argparse.ArgumentParser()
    #注意文件路径
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
	

    #模型定义
    graph = tf.Graph()
    with graph.as_default():
        #定义训练数据
        input_data = tf.placeholder(tf.int32, [args.batch_size, args.win_size])
        targets = tf.placeholder(tf.int64, [args.batch_size, 1])
		
        #模型参数
        with tf.variable_scope('nnlm' + 'embedding'):
            embeddings = tf.Variable(tf.random_uniform([args.vocab_size, args.word_dim], -1.0, 1.0))#均匀分布
            embeddings = tf.nn.l2_normalize(embeddings, 1)

        with tf.variable_scope('nnlm' + 'weight'):
            # tf.truncated_normal()从截断的正态分布中输出随机值,shape表示生成张量的维度，mean是均值，stddev是标准差。
            '''weight_h = tf.Variable(tf.truncated_normal([args.win_size * args.word_dim + 1, args.hidden_num],
                            stddev=1.0 / math.sqrt(args.hidden_num))) 
           
            softmax_w = tf.Variable(tf.truncated_normal([args.win_size * args.word_dim, args.vocab_size],
                            stddev=1.0 / math.sqrt(args.win_size * args.word_dim)))
            softmax_u = tf.Variable(tf.truncated_normal([args.hidden_num + 1, args.vocab_size],
                            stddev=1.0 / math.sqrt(args.hidden_num)))'''
            '''weight_h = tf.Variable(tf.truncated_normal([args.win_size, args.hidden_num],
                            stddev=1.0 / math.sqrt(args.hidden_num)))
            softmax_w = tf.Variable(tf.truncated_normal([args.win_size, args.vocab_size], stddev=1.0 / math.sqrt(args.win_size * args.word_dim)))
            softmax_u = tf.Variable(tf.truncated_normal([args.hidden_num, args.vocab_size],
                                                        stddev=1.0 / math.sqrt(args.hidden_num)))'''
            weight_h = tf.Variable(tf.truncated_normal([args.win_size * args.word_dim, args.hidden_num],
                                                       stddev=1.0 / math.sqrt(args.hidden_num)))
            softmax_w = tf.Variable(tf.truncated_normal([args.win_size * args.word_dim, args.vocab_size],
                                                        stddev=1.0 / math.sqrt(args.win_size * args.word_dim)))
            softmax_u = tf.Variable(tf.truncated_normal([args.hidden_num, args.vocab_size],
                                                        stddev=1.0 / math.sqrt(args.hidden_num)))

        with tf.variable_scope('nnlm'+'bias'):
            b1 = tf.Variable(tf.random_normal([args.hidden_num]))
            b2 = tf.Variable(tf.random_normal([args.vocab_size]))

        #TODO，构造计算图
        def infer_output(input_data):
            #NNLM
            '''input_data=tf.cast(input_data,dtype=tf.float32)
            hidden = tf.nn.tanh(tf.matmul(input_data , weight_h) + b1)
            outputs = tf.nn.softmax(tf.matmul(input_data ,softmax_w) + tf.matmul(hidden ,softmax_u) + b2)'''
            # tf.nn.embedding_lookup函数的用法主要是选取一个张量里面索引对应的元素。
            # tf.nn.embedding_lookup（params, ids）:params可以是张量也可以是数组等，id就是对应的索引.
            input_data_emb0 = tf.nn.embedding_lookup(embeddings, input_data)
            input_data_emb = tf.reshape(input_data_emb0, [-1, args.win_size * args.word_dim])  #-1是不指定该维，程序自动计算 这里就是args.batch_size
            hidden = tf.tanh(tf.matmul(input_data_emb, weight_h)) + b1
            hidden_output = tf.matmul(hidden, softmax_u) + tf.matmul(input_data_emb, softmax_w) + b2
            outputs = tf.nn.softmax(hidden_output)
            return outputs,input_data_emb0,input_data_emb

        outputs,input_data_emb0,input_data_emb = infer_output(input_data)
        one_hot_targets = tf.one_hot(tf.squeeze(targets), args.vocab_size, 1.0, 0.0)#这里为什么要squeeze()??squeeze()给定张量输入，此操作返回相同类型的张量，并删除所有尺寸为1的尺寸

        loss = -tf.reduce_mean(tf.reduce_sum(tf.log(outputs) * one_hot_targets, 1)) #对数损失函数

        #optimizer = tf.train.AdagradOptimizer(0.1).minimize(loss)

        # Clip grad.解决梯度爆炸问题
        optimizer = tf.train.AdagradOptimizer(0.1)
        gvs = optimizer.compute_gradients(loss)
        #tf.clip_by_value(A, min, max)：输入一个张量A，把A中的每一个元素的值都压缩在min和max之间。小于min的让它等于min，大于max的元素的值等于max。
        capped_gvs = [(tf.clip_by_value(grad, -args.grad_clip, args.grad_clip), var) for grad, var in gvs]
        optimizer = optimizer.apply_gradients(capped_gvs)


        #输出词向量    square平方操作
        embeddings_norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), axis=1, keep_dims=True))#axis=1:计算每一行的和
        normalized_embeddings = embeddings / embeddings_norm
        #print "normalized_embeddings:",normalized_embeddings

    #模型训练
    with tf.Session(graph=graph) as sess:
        tf.global_variables_initializer().run()
        for e in range(args.num_epochs):
            data_loader.reset_batch_pointer()
            for b in range(data_loader.num_batches):
                start = time.time()
                x, y = data_loader.next_batch()
                feed = {input_data: x, targets: y}
                train_loss,  _ = sess.run([loss, optimizer], feed)
                end = time.time()
                print "prediction:",outputs
                print "label:",one_hot_targets
                print "input_data_emb0:",input_data_emb0
                print "input_data_emb:", input_data_emb
                print("{}/{} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}" .format(b, data_loader.num_batches,e, train_loss, end - start))
			
			# 保存词向量至nnlm_word_embeddings.npy文件
            np.save('nnlm_word_embeddings.zh', normalized_embeddings.eval()) #eval()可以将字符串转化成表达式
            #np.save('nnlm_word_embeddings.en', normalized_embeddings.eval())

        

if __name__ == '__main__':
    main()
