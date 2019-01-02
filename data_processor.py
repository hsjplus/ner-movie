# -*- coding: utf-8 -*-

from konlpy.tag import Okt
import pickle
import re
import tensorflow as tf

class Processor:
  def __init__(self):
    self.max_sent_len = 16
    # a set of indexes to remove sentences with ambiguous labelling problems   
    self.remove_sentence_idx = set()
    
  # tokenize data and put pos-tagging on data
  def tokenize_pos_data(self, data):
    okt = Okt()

    tokenized_data = []
    pos_data = []

    for sent in data: 
      tokenized_data.append([word for word, pos in okt.pos(sent)])
      pos_data.append([pos for word, pos in okt.pos(sent)])

    return (tokenized_data, pos_data)
  
  # load domain dictionary list
  def load_domain_dictionary_list(self, directory_path):
    # movie
    with open(directory_path + 'movie.pickle', 'rb') as f:
      movie = pickle.load(f)

    # actor
    with open(directory_path + 'actor.pickle', 'rb') as f:
      actor = pickle.load(f)

    # director
    with open(directory_path + 'director.pickle', 'rb') as f:
      director = pickle.load(f)

    # genre
    with open(directory_path + 'genre_B.pickle', 'rb') as f:
      genre = pickle.load(f)

    # country
    with open(directory_path + 'country_B.pickle', 'rb') as f:
      country = pickle.load(f)

    domain_dict_list = [movie, actor, director, genre, country]

    return domain_dict_list

  # generate labels
  # 0: movie, 1: actor, 2: director, 3: genre, 4: country, 5: year, 6: country, 7: others
  def generate_labels(self, tokenized_data):
    # make labels with sentence data by one-hot encoding method
    label = []
    # a set of indexes to remove sentences with ambiguous labelling problems   
    idx = 0 
    for sentence in tokenized_data:
      sentence_idx = []
      for word in sentence:
        word_idx = []
        cnt = 0
        # movie, actor, director, genre, country
        for one_dict in self.load_domain_dictionary_list('data/'):    
          if word in one_dict:
            word_idx.append(1)
            cnt +=1
          else:
            word_idx.append(0)
        # year 
        year_regex = re.compile('\d+년|[12]\d{3}')
        if len(year_regex.findall(word)) == 1:
          word_idx.append(1)
          cnt +=1
        else:
          word_idx.append(0)

        # month
        month_regex = re.compile('\d+월')  
        if len(month_regex.findall(word)) == 1:
          word_idx.append(1)
          cnt += 1
        else:
          word_idx.append(0)

        # others
        if cnt == 0:
          word_idx.append(1)
        else:
          word_idx.append(0)

        if cnt > 1:
          self.remove_sentence_idx.add(idx)

        sentence_idx.append(word_idx)
      label.append(sentence_idx)
      idx += 1 

    # transform labels with one-hot encoding to index number
    indexed_label = []
    for sent in label:
      tmp = []
      for word in sent:
        tmp.append(word.index(1))
      indexed_label.append(tmp)
    
    # put paddings on every label
    others_idx = 7
    indexed_label = tf.keras.preprocessing.sequence.pad_sequences(indexed_label, maxlen=self.max_sent_len,
                                                                  padding='post', value=others_idx, dtype='int64')
    
    return indexed_label
  
  # remove ambiguous data
  def remove_ambiguous_data(self, tokenized_data, pos_data, indexed_label):
    remained_sentence_idx = set(range(len(tokenized_data)))
    for idx in self.remove_sentence_idx:
      remained_sentence_idx.remove(idx)

    tokenized_data = [tokenized_data[i] for i in remained_sentence_idx]
    pos_data = [pos_data[i] for i in remained_sentence_idx]
    indexed_label = [indexed_label[i] for i in remained_sentence_idx]
    
    return (tokenized_data, pos_data, indexed_label)