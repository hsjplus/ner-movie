# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from data_loader import load_data
from data_processor import Processor
from embedding_builder import Embedding
from model import Model

import warnings
warnings.simplefilter('ignore')

if __name__ == '__main__':    
  # load data
  data = load_data('data/intent_v3.txt')

  # set parameter 
  parameter = {'max_sent_len': 16, 'max_word_len': 12, 'batch_size': 100, 'total_embedding_size': 238, 'class_num': 8,
              'lstm_units': 128, 'syl_lstm_units': 8, 'learning_rate': 0.01, 'epochs': 3}

  # process data
  data_processor = Processor()
  tokenized_data, pos_data = data_processor.tokenize_pos_data(data)
  indexed_label = data_processor.generate_labels(tokenized_data)
  tokenized_data, pos_data, indexed_label = data_processor.remove_ambiguous_data(tokenized_data, pos_data, indexed_label)
  x_train, x_test, x_train_pos, x_test_pos, y_train, y_test = train_test_split(tokenized_data, pos_data, indexed_label, test_size=0.2, random_state=0)

  # make concatenated-input-embedding 
  domain_dict_list = data_processor.load_domain_dictionary_list('data/')
  model = Model(parameter)
  embedding = Embedding(domain_dict_list, model, parameter)
  concat_embedding = embedding.make_concat_embedding(x_train, x_train_pos)
  concat_embedding_test = embedding.make_concat_embedding(x_test, x_test_pos)

  # train!
  model.build_total_model()

  sess = tf.Session()
  sess.run(tf.global_variables_initializer())

  epochs = parameter['epochs']
  total_batch = int(concat_embedding.shape[0]/parameter['batch_size'])

  for epoch in range(epochs):
    loss_list = [] 
    
    for i in range(total_batch):
      x_batch = concat_embedding[parameter['batch_size']*i:parameter['batch_size']*(i+1)]
      y_batch = y_train[parameter['batch_size']*i:parameter['batch_size']*(i+1)]
      
      _, loss = sess.run([model.optimizer, model.cost],
                        feed_dict={model.X: x_batch, model.Y: y_batch})
      loss_list.append(loss)
      
    loss_average = np.array(loss_list).mean()
    print('Epoch:', '{} / {}'.format(epoch+1, epochs),'=> loss = {:.4f}'.format(loss_average))
    
  # accuracy on test data
  total_sum = 0
  true_sum = 0 
  total_batch = int(concat_embedding_test.shape[0]/parameter['batch_size'])

  for i in range(total_batch):
    x_batch = concat_embedding[parameter['batch_size']*i:parameter['batch_size']*(i+1)]
    prediction_ = sess.run(model.prediction, feed_dict={model.X: x_batch})
    y_batch = y_train[parameter['batch_size']*i:parameter['batch_size']*(i+1)]
    
    # compare with prediction and real labels
    true_sum += (y_batch == prediction_).sum()
    total_sum += parameter['max_sent_len'] * parameter['batch_size'] 
    
  print('Test Accuracy: {:.4f}'.format(true_sum / total_sum))