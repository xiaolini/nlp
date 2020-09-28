#encoding:utf-8
import numpy as np
import os, time, sys
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell
from tensorflow.contrib.crf import crf_log_likelihood
from tensorflow.contrib.crf import viterbi_decode
from data import pad_sequences, batch_yield
from utils import get_logger  #写日志
from eval import conlleval  #在命名体识别任务（NER，Named Entity Recognizer）中,Evaluate使用Perl的conlleval.pl


class BiLSTM_CRF(object):
    def __init__(self, args, embeddings, tag2label, vocab, paths, config):
        self.batch_size = args.batch_size
        self.epoch_num = args.epoch
        self.hidden_dim = args.hidden_dim
        self.embeddings = embeddings
        self.CRF = args.CRF  #最后一层是否使用CRF
        self.update_embedding = args.update_embedding
        self.dropout_keep_prob = args.dropout
        self.optimizer = args.optimizer
        self.lr = args.lr
        self.clip_grad = args.clip
        self.tag2label = tag2label
        self.num_tags = len(tag2label)
        self.vocab = vocab
        self.shuffle = args.shuffle
        self.model_path = paths['model_path']
        self.summary_path = paths['summary_path']
        self.logger = get_logger(paths['log_path'])
        self.result_path = paths['result_path']
        self.config = config

    def build_graph(self):
        self.add_placeholders()
        self.lookup_layer_op()
        self.biLSTM_layer_op()
        self.softmax_pred_op()
        self.loss_op()
        self.trainstep_op()
        self.init_op()

    def add_placeholders(self):
        self.word_ids = tf.placeholder(tf.int32, shape=[None, None], name="word_ids")
        #真实标签
        self.labels = tf.placeholder(tf.int32, shape=[None, None], name="labels")
        #真实句子长度
        self.sequence_lengths = tf.placeholder(tf.int32, shape=[None], name="sequence_lengths")

        self.dropout_pl = tf.placeholder(dtype=tf.float32, shape=[], name="dropout")
        self.lr_pl = tf.placeholder(dtype=tf.float32, shape=[], name="lr")

    def lookup_layer_op(self):
        with tf.variable_scope("words"):
            _word_embeddings = tf.Variable(self.embeddings,
                                           dtype=tf.float32,
                                           trainable=self.update_embedding,
                                           name="_word_embeddings")
            word_embeddings = tf.nn.embedding_lookup(params=_word_embeddings,
                                                     ids=self.word_ids,
                                                     name="word_embeddings")
        self.word_embeddings = tf.nn.dropout(word_embeddings, self.dropout_pl)

    def biLSTM_layer_op(self):
        with tf.variable_scope("bi-lstm"):
            cell_fw = LSTMCell(self.hidden_dim) #隐藏层神经元，默认300
            cell_bw = LSTMCell(self.hidden_dim)
            # def bidirectional_dynamic_rnn(
            #         cell_fw,  # 前向RNN
            #         cell_bw,  # 后向RNN
            #         inputs,  # 输入
            #         sequence_length=None,  # 输入序列的实际长度（可选，默认为输入序列的最大长度）
            #         initial_state_fw=None,  # 前向的初始化状态（可选）
            #         initial_state_bw=None,  # 后向的初始化状态（可选）
            #         dtype=None,  # 初始化和输出的数据类型（可选）
            #         parallel_iterations=None,
            #         swap_memory=False,
            #         time_major=False,
            #         # 决定了输入输出tensor的格式：如果为true, 向量的形状必须为 `[max_time, batch_size, depth]`.
            #         # 如果为false, tensor的形状必须为`[batch_size, max_time, depth]`.
            #         scope=None
            # )
            # outputs为(output_fw, output_bw)，是一个包含前向cell输出tensor和后向cell输出tensor组成的二元组。
            # 如果time_major == False(默认值), 则output_fw将是形状为[batch_size, max_time, cell_fw.output_size]的张量, 则output_bw将是形状为[batch_size, max_time, cell_bw.output_size]的张量；
            # 如果time_major == True, 则output_fw将是形状为[max_time, batch_size, cell_fw.output_size]的张量；output_bw将会是形状为[max_time, batch_size, cell_bw.output_size]的张量.
            # 与bidirectional_rnn不同, 它返回一个元组而不是单个连接的张量.如果优选连接的, 则正向和反向输出可以连接为tf.concat(outputs, 2).

            # output_states为(output_state_fw, output_state_bw)，包含了前向和后向最后的隐藏状态的组成的二元组。 
            # output_state_fw和output_state_bw的类型为LSTMStateTuple。 
            # LSTMStateTuple由（c，h）组成，分别代表memory cell和hidden（即c,h矩阵）state。

            (output_fw_seq, output_bw_seq), _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell_fw,
                cell_bw=cell_bw,
                inputs=self.word_embeddings,
                sequence_length=self.sequence_lengths,
                dtype=tf.float32)
            output = tf.concat([output_fw_seq, output_bw_seq], axis=-1)
            output = tf.nn.dropout(output, self.dropout_pl)

        with tf.variable_scope("proj"):
            #tf.get_variable(<name>, <shape>, <initializer>) 创建或返回给定名称的变量
            W = tf.get_variable(name="W",
                                shape=[2 * self.hidden_dim, self.num_tags],
                                initializer=tf.contrib.layers.xavier_initializer(),
                                dtype=tf.float32)

            b = tf.get_variable(name="b",
                                shape=[self.num_tags],
                                initializer=tf.zeros_initializer(),
                                dtype=tf.float32)

            s = tf.shape(output)
            output = tf.reshape(output, [-1, 2*self.hidden_dim])
            pred = tf.matmul(output, W) + b

            self.logits = tf.reshape(pred, [-1, s[1], self.num_tags])   #预测得到的标签结果

    def loss_op(self):
        if self.CRF:

            # crf_log_likelihood作为损失函数
            # inputs：unary potentials,就是每个标签的预测概率值
            # tag_indices，这个就是真实的标签序列了
            # sequence_lengths,一个样本真实的序列长度，为了对齐长度会做些padding，但是可以把真实的长度放到这个参数里
            # transition_params,转移概率，可以没有，没有的话这个函数也会算出来
            # 输出：log_likelihood:标量;transition_params,转移概率，如果输入没输，它就自己算个给返回

            #为了让预测标签和真实标签更加相似，使用crf最大似然self.logits是预测值，self.labels是真实值是通过feed_dict输入的
            log_likelihood, self.transition_params = crf_log_likelihood(inputs=self.logits,
                                                                   tag_indices=self.labels,
                                                                   sequence_lengths=self.sequence_lengths)
            #transition_params形状为[num_tags, num_tags] 的转移矩阵
            self.loss = -tf.reduce_mean(log_likelihood)

        else:
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits,
                                                                   labels=self.labels)
            #张量变换函数
            #sequence_mask(
             #   lengths,
              #  maxlen=None,
               # dtype=tf.bool,
                #name=None
            #)
            #lengths：整数张量，其所有值小于等于maxlen。
            #maxlen：标量整数张量，返回张量的最后维度的大小；默认值是lengths中的最大值。
            #dtype：结果张量的输出类型。
            #name：操作的名字。
            #eg:a = tf.sequence_mask([1, 2, 3], 5)输出：
             #[[ True False False False False]
              #  [ True  True False False False]
              #  [ True  True  True False False]]
            #解析：maxlen是5，所以一共有5列，lengths有三个元素[1,2,3]，所以有三行，每一行分别前1、2、3个元素为True

            mask = tf.sequence_mask(self.sequence_lengths)
            #boolean_mask(a,b) 将使a (m维)矩阵仅保留与b中“True”元素同下标的部分的值，然后求平均值
            losses = tf.boolean_mask(losses, mask)
            self.loss = tf.reduce_mean(losses)
        #添加标量统计结果
        tf.summary.scalar("loss", self.loss)

    def softmax_pred_op(self):
        if not self.CRF:
            self.labels_softmax_ = tf.argmax(self.logits, axis=-1)
            self.labels_softmax_ = tf.cast(self.labels_softmax_, tf.int32)

    def trainstep_op(self):
        with tf.variable_scope("train_step"):
            self.global_step = tf.Variable(0, name="global_step", trainable=False)
            if self.optimizer == 'Adam':
                optim = tf.train.AdamOptimizer(learning_rate=self.lr_pl)
            elif self.optimizer == 'Adadelta':
                optim = tf.train.AdadeltaOptimizer(learning_rate=self.lr_pl)
            elif self.optimizer == 'Adagrad':
                optim = tf.train.AdagradOptimizer(learning_rate=self.lr_pl)
            elif self.optimizer == 'RMSProp':
                optim = tf.train.RMSPropOptimizer(learning_rate=self.lr_pl)
            elif self.optimizer == 'Momentum':
                optim = tf.train.MomentumOptimizer(learning_rate=self.lr_pl, momentum=0.9)
            elif self.optimizer == 'SGD':
                optim = tf.train.GradientDescentOptimizer(learning_rate=self.lr_pl)
            else:
                optim = tf.train.GradientDescentOptimizer(learning_rate=self.lr_pl)

            # minimize()实际上包含了两个步骤，即compute_gradients和apply_gradients，前者用于计算梯度，后者用于使用计算得到的梯度来更新对应的variable
            grads_and_vars = optim.compute_gradients(self.loss)
            #clip操作防止梯度爆炸
            grads_and_vars_clip = [[tf.clip_by_value(g, -self.clip_grad, self.clip_grad), v] for g, v in grads_and_vars]
            self.train_op = optim.apply_gradients(grads_and_vars_clip, global_step=self.global_step)

    def init_op(self):
        self.init_op = tf.global_variables_initializer()

    def add_summary(self, sess):  #tf.summary()保存训练过程以及参数分布图并在tensorboard显示
        """
        :param sess:
        :return:
        """
        self.merged = tf.summary.merge_all()  #将所有summary全部保存到磁盘，以便tensorboard显示。如果没有特殊要求，一般用这一句就可一显示训练时的各种信息了。
        self.file_writer = tf.summary.FileWriter(self.summary_path, sess.graph) #指定一个文件用来保存图。

    def train(self, train, dev):
        """
        :param train:
        :param dev:
        :return:
        """
        saver = tf.train.Saver(tf.global_variables())  #将训练好的模型参数保存起来，以便以后进行验证或测试。tf.global_variables()查看管理变量的函数

        with tf.Session(config=self.config) as sess:
            sess.run(self.init_op)
            self.add_summary(sess)

            for epoch in range(self.epoch_num):
                self.run_one_epoch(sess, train, dev, self.tag2label, epoch, saver)

    def test(self, test):
        saver = tf.train.Saver()
        with tf.Session(config=self.config) as sess:
            self.logger.info('=========== testing ===========')
            saver.restore(sess, self.model_path)  #加载模型
            label_list, seq_len_list = self.dev_one_epoch(sess, test)
            self.evaluate(label_list, seq_len_list, test)

    def demo_one(self, sess, sent):
        """
        :param sess:
        :param sent:
        :return:
        """
        label_list = []
        for seqs, labels in batch_yield(sent, self.batch_size, self.vocab, self.tag2label, shuffle=False):
            label_list_, _ = self.predict_one_batch(sess, seqs)
            label_list.extend(label_list_)
        label2tag = {}
        for tag, label in self.tag2label.items():
            label2tag[label] = tag if label != 0 else label
        tag = [label2tag[label] for label in label_list[0]]
        return tag

    def run_one_epoch(self, sess, train, dev, tag2label, epoch, saver):
        """
        :param sess:
        :param train:
        :param dev:
        :param tag2label:
        :param epoch:
        :param saver:
        :return:
        """
        num_batches = (len(train) + self.batch_size - 1) // self.batch_size

        start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        batches = batch_yield(train, self.batch_size, self.vocab, self.tag2label, shuffle=self.shuffle)
        for step, (seqs, labels) in enumerate(batches):

            sys.stdout.write(' processing: {} batch / {} batches.'.format(step + 1, num_batches) + '\r')
            step_num = epoch * num_batches + step + 1
            feed_dict, _ = self.get_feed_dict(seqs, labels, self.lr, self.dropout_keep_prob)  #输入数据
            _, loss_train, summary, step_num_ = sess.run([self.train_op, self.loss, self.merged, self.global_step],
                                                         feed_dict=feed_dict)
            if step + 1 == 1 or (step + 1) % 300 == 0 or step + 1 == num_batches:
                self.logger.info(
                    '{} epoch {}, step {}, loss: {:.4}, global_step: {}'.format(start_time, epoch + 1, step + 1,
                                                                                loss_train, step_num))

            self.file_writer.add_summary(summary, step_num)

            if step + 1 == num_batches:
                saver.save(sess, self.model_path, global_step=step_num)  #保存模型，名字为self.model_path-global_step，恢复模型：saver.restore()

        self.logger.info('===========validation / test===========')
        label_list_dev, seq_len_list_dev = self.dev_one_epoch(sess, dev)
        self.evaluate(label_list_dev, seq_len_list_dev, dev, epoch)

    # 占位符赋值
    def get_feed_dict(self, seqs, labels=None, lr=None, dropout=None):
        """
        :param seqs:
        :param labels:
        :param lr:
        :param dropout:
        :return: feed_dict
        """
        word_ids, seq_len_list = pad_sequences(seqs, pad_mark=0)

        feed_dict = {self.word_ids: word_ids,
                     self.sequence_lengths: seq_len_list}
        if labels is not None:
            # labels经过padding后，喂给feed_dict
            labels_, _ = pad_sequences(labels, pad_mark=0)
            feed_dict[self.labels] = labels_
        if lr is not None:
            feed_dict[self.lr_pl] = lr
        if dropout is not None:
            feed_dict[self.dropout_pl] = dropout

        return feed_dict, seq_len_list

    def dev_one_epoch(self, sess, dev):
        """
        :param sess:
        :param dev:
        :return:
        """
        label_list, seq_len_list = [], []
        for seqs, labels in batch_yield(dev, self.batch_size, self.vocab, self.tag2label, shuffle=False):
            label_list_, seq_len_list_ = self.predict_one_batch(sess, seqs)
            label_list.extend(label_list_)
            seq_len_list.extend(seq_len_list_)
        return label_list, seq_len_list

    def predict_one_batch(self, sess, seqs):
        """
        :param sess:
        :param seqs:
        :return: label_list
                 seq_len_list
        """
        feed_dict, seq_len_list = self.get_feed_dict(seqs, dropout=1.0)

        if self.CRF:
            logits, transition_params = sess.run([self.logits, self.transition_params],
                                                 feed_dict=feed_dict)
            label_list = []
            # zip()打包成元素形式为元组的列表[(logit,seq_len),(logit,seq_len),( ,),]
            for logit, seq_len in zip(logits, seq_len_list):#zip()函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表。以最短的为标准，长的截断
                # 如果是demo情况下，输入句子，那么只有一个句子，所以只循环一次，训练模式下就不会对logits解析得到一个数
                viterbi_seq, _ = viterbi_decode(logit[:seq_len], transition_params)#在predict_one_batch进行预测的时候要将biLSTM的到的结果进行viterbi解码，从而得到最优序列。
            # 对logit按行解析返回的值[[1, 2, 0, 0, 0, 0, 3, 4, 0, 5, 6, 6, 6]]#这就是预测结果，对应着tag2label里的值
                label_list.append(viterbi_seq)
            return label_list, seq_len_list

        else: #如果不用CRF，就是把self.logits每行取最大的
            label_list = sess.run(self.labels_softmax_, feed_dict=feed_dict)
            return label_list, seq_len_list

    def evaluate(self, label_list, seq_len_list, data, epoch=None):
        """
        :param label_list:
        :param seq_len_list:
        :param data:
        :param epoch:
        :return:
        """
        label2tag = {}
        for tag, label in self.tag2label.items():
            label2tag[label] = tag if label != 0 else label

        model_predict = []
        for label_, (sent, tag) in zip(label_list, data):
            tag_ = [label2tag[label__] for label__ in label_]
            sent_res = []
            if  len(label_) != len(sent):
                print(sent)
                print(len(label_))
                print(tag)
            for i in range(len(sent)):
                sent_res.append([sent[i], tag[i], tag_[i]])
            model_predict.append(sent_res)
        epoch_num = str(epoch+1) if epoch != None else 'test'
        label_path = os.path.join(self.result_path, 'label_' + epoch_num)
        metric_path = os.path.join(self.result_path, 'result_metric_' + epoch_num)
        for _ in conlleval(model_predict, label_path, metric_path):
            self.logger.info(_)