# -*- coding: utf-8 -*-

import tensorflow as tf

class Model:
  def __init__(self, parameter):
    self.parameter = parameter
  
  # build syllable embedding related bi-lstm model
  def build_syllable_model(self, syl2idx):
    with tf.variable_scope('syl', reuse=tf.AUTO_REUSE):
      sent_size = 1
      syl_size = len(syl2idx) # 8159

      self.syl_X = tf.placeholder(tf.float64, [sent_size, self.parameter['max_word_len'], syl_size]) # (1, 12, 8159)
      # bi-lstm cell 
      syl_fw_cell = tf.contrib.rnn.LSTMCell(self.parameter['syl_lstm_units'])
      syl_bw_cell = tf.contrib.rnn.LSTMCell(self.parameter['syl_lstm_units'])
      _, self.syl_states = tf.nn.bidirectional_dynamic_rnn(syl_fw_cell, syl_bw_cell, self.syl_X, dtype=tf.float64)
    
    self.sess = tf.Session()
    saver = tf.train.Saver(tf.global_variables(scope='syl'))
    saver.restore(self.sess, 'syl_lstm/syl_lstm.model')
    
  def run_syllable_model(self, x_input):  
    syl_states_ = self.sess.run(self.syl_states, feed_dict={self.syl_X: x_input})
    
    return syl_states_
  
  # build concatenated-input-embedding related bi-lstm model
  def build_total_model(self):
    with tf.variable_scope('total', reuse=tf.AUTO_REUSE):
      self.X = tf.placeholder(tf.float64, [self.parameter['batch_size'], self.parameter['max_sent_len'], self.parameter['total_embedding_size']])
      self.Y = tf.placeholder(tf.int64, [None, None])

      # weight 
      W = tf.Variable(tf.random_normal([self.parameter['lstm_units'], self.parameter['class_num']], dtype=tf.float64))
      b = tf.Variable(tf.random_normal([self.parameter['class_num']], dtype=tf.float64))

      # bi-LSTM 
      fw_cell = tf.contrib.rnn.LSTMCell(self.parameter['lstm_units'])
      bw_cell = tf.contrib.rnn.LSTMCell(self.parameter['lstm_units'])
      outputs, states = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, self.X, dtype=tf.float64) # (batch_size, max_sent_len, lstm_units)
      outputs_reshaped = tf.reshape(outputs[1], (-1, self.parameter['lstm_units'])) # (batch_size * max_sent_len, lstm_units)

      # softmax layer
      logits = tf.matmul(outputs_reshaped, W) + b
      logits = tf.reshape(logits, (self.parameter['batch_size'], self.parameter['max_sent_len'], self.parameter['class_num']))
      self.cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=self.Y))
      
      self.optimizer = tf.train.AdamOptimizer(self.parameter['learning_rate']).minimize(self.cost)
      self.prediction = tf.cast(tf.argmax(logits, 2), tf.int64)
  
  def close_session(self):
    self.sess.close()