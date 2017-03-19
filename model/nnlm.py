# -*- coding:utf-8 -*-
import numpy as np
import tensorflow as tf
import math

"""
implementing "A neural probabilistic language model, Yoshua Bengio, 2003" on tensorflow
"""
class NNLM(object):
    def __init__(self, config, sess):

        self.batch_size = config.batch_size
        self.num_epochs = config.num_epochs

        self.nwords = config.nwords
        self.win_size = config.win_size
        self.hidden_num = config.hidden_num
        self.word_dim = config.word_dim
        self.grad_clip = config.grad_clip

        self.input = tf.placeholder(tf.int32, [self.batch_size, self.win_size])
        self.targets = tf.placeholder(tf.float32, [self.batch_size,  self.nwords])

        self.sess = sess
        self.is_test = config.is_test
        self.show = config.show

    def build_model(self):
        #embeddings
        self.C = tf.Variable(tf.random_uniform([self.nwords, self.word_dim], -1.0, 1.0)) # nwords * word_dim
        self.C = tf.nn.l2_normalize(self.C, 1)

        #embed2hidden 
        self.H = tf.Variable(tf.truncated_normal([self.win_size * self.word_dim + 1, self.hidden_num], stddev=1.0/math.sqrt(self.hidden_num))) #h * (word_dim * win_size), encoding d matrix with adding 1?
#        self.d = tf.Variable()

        self.W = tf.Variable(tf.truncated_normal([self.win_size * self.word_dim, self.nwords], stddev=1.0/math.sqrt(self.win_size * self.word_dim) ))
        self.U = tf.Variable(tf.truncated_normal([self.hidden_num + 1, self.nwords], stddev=1.0/math.sqrt(self.hidden_num))) # why add 1?
#        self.b = tf.Variable() #encoding in self.U? 

        input_embeds = tf.nn.embedding_lookup(self.C, self.input)
        input_embeds = tf.reshape(input_embeds, [-1, self.win_size * self.word_dim])
        b_tmp = tf.stack([tf.shape(self.input)[0],1])
        b = tf.ones(b_tmp)
        input_embeds_add = tf.concat([input_embeds, b],1)

        hidden_out = tf.tanh(tf.matmul(input_embeds_add, self.H))
        hidden_out_add = tf.concat([hidden_out, tf.ones(tf.stack([tf.shape(self.input)[0], 1]))], 1)

        output = tf.matmul(hidden_out_add, self.U) + tf.matmul(input_embeds, self.W)
        output = tf.clip_by_value(output, 0.0, self.grad_clip)

        output = tf.nn.softmax(output)

        #loss function & optimization algorithm
        self.loss = -tf.reduce_mean(tf.reduce_sum(tf.log(output) * self.targets, 1))
        self.optim = tf.train.AdagradOptimizer(0.1).minimize(self.loss) # what kind of the optimizer it is?

        tf.global_variables_initializer().run()

    def train(self, data):
        N = int(math.ceil(len(data) / self.batch_size))
        cost = 0

        x = np.ndarray([self.batch_size, self.win_size], dtype=np.float32)
        target = np.zeros([self.batch_size, self.nwords]) # one-hot-encoded

        if self.show:
            from utils import ProgressBar
            bar = ProgressBar('Train', max=N)

        m = self.win_size;
        clean_data = np.concatenate((np.zeros(self.win_size, dtype=np.int32), data)) #padding head
        clean_data = np.concatenate((clean_data, np.zeros(self.batch_size, dtype=np.int32))) #padding tail
        print("original data head=%s" % (data[0:10]))
        print("clean_data head=%s" % (clean_data[0:10]))
        for idx in xrange(N): #interations
            if self.show: bar.next()
            target.fill(0)
            for b in xrange(self.batch_size):
                target[b][clean_data[m]] = 1
                m += 1

                x[b] = data[m-self.win_size : m] # we need padding here!

            _, loss = self.sess.run([self.optim, self.loss], feed_dict={
                                                        self.input: x,
                                                        self.targets: target})
            cost += np.sum(loss) 

        if self.show: bar.finish()
        return cost / N /self.batch_size

    def run(self, train_data, test_data):
        if not self.is_test:
            for e in range(self.num_epochs):
                train_loss = np.sum(self.train(train_data))
                state = {
                    "perplexity":math.exp(train_loss),
                    "epoch":e
                }
                print state
        else :
            print "testing"
