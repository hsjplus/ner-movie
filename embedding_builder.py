# -*- coding: utf-8 -*-

from gensim.models import word2vec
import numpy as np
import tensorflow as tf
import re
import pickle

class Embedding:
  def __init__(self, domain_dict_list, model, parameter):
    self.parameter = parameter
    self.model = model 
    
    # load pretrained corpus word2vec model
    WIKI_MODEL_PATH = 'data/wiki.model'
    MOVIE_MODEL_PATH = 'data/2012movie.model'
    
    wiki_model = word2vec.Word2Vec.load(WIKI_MODEL_PATH)
    movie_model = word2vec.Word2Vec.load(MOVIE_MODEL_PATH)

    # wiki 
    self.wiki_word2idx = {word: i for i, word in enumerate(wiki_model.wv.index2word)}
    self.wiki_matrix = wiki_model.wv.vectors
    
    # movie 
    self.movie_word2idx = {word: i for i, word in enumerate(movie_model.wv.index2word)}
    self.movie_matrix = movie_model.wv.vectors

    # pos2idx 
    self.pos2idx = {'Adjective': 5, 'Adverb': 10, 'Alpha': 11, 'Determiner': 6, 'Eomi': 0,
                    'Exclamation': 7, 'Foreign': 8, 'Josa': 1, 'Modifier': 9, 'Noun': 13,
                    'Number': 4, 'Punctuation': 12, 'Suffix': 3, 'Verb': 14, 'VerbPrefix': 2}
    
    # domain_dict_list
    self.domain_dict_list = domain_dict_list
    
    # syl2idx
    SYL2IDX_PATH = 'data/syl2idx.pickle'
    with open(SYL2IDX_PATH, 'rb') as f:
      self.syl2idx = pickle.load(f)

  # (1) make word2vec embeddings - wiki corpus, movie domain corpus  
  def make_word2vec_embedding(self, x_train, corpus_type):
    # use diffent corpus depending on corpus types
    if corpus_type == 'wiki':
      word2idx = self.wiki_word2idx
      matrix = self.wiki_matrix
    elif corpus_type == 'movie':
      word2idx = self.movie_word2idx
      matrix = self.movie_matrix
    
    X_train = []
    X_train_vect = []
    max_idx = len(word2idx)
    zeros = np.zeros([100])
    
    # word -> idx
    for sent in x_train:
      tmp = []
      for word in sent:
        try:
          tmp.append(word2idx[word])
        except:
          tmp.append(max_idx)
      X_train.append(tmp)

    X_train = tf.keras.preprocessing.sequence.pad_sequences(X_train, maxlen=self.parameter['max_sent_len'], padding='post', value=max_idx)
    
    # idx -> vectors
    for idx_list in X_train:
      tmp = []
      for idx in idx_list: 
        try:
          tmp.append(self.wiki_matrix[idx])
        except:
          tmp.append(zeros)
      X_train_vect.append(tmp)

    return np.array(X_train_vect)
  
  # (2) make POS embedding by one-hot encoding way
  def make_pos_embedding(self, x_train_pos):

    max_idx = len(self.pos2idx)
    X_train_pos = []
    X_train_pos_vect = []
    pos_matrix = np.eye(max_idx)
    pos_zeros = np.zeros([max_idx])
    
    # pos -> idx
    for sent in x_train_pos:
      tmp = []
      for pos in sent:
        tmp.append(self.pos2idx[pos])
      X_train_pos.append(tmp)

    X_train_pos = tf.keras.preprocessing.sequence.pad_sequences(X_train_pos, maxlen=self.parameter['max_sent_len'], padding='post', value=max_idx)
    
    # idx -> one-hot vectors
    for idx_list in X_train_pos:
      tmp = []
      for idx in idx_list: 
        try:
          tmp.append(pos_matrix[idx])
        except:
          tmp.append(pos_zeros)
      X_train_pos_vect.append(tmp)

    return np.array(X_train_pos_vect)
  
  # (3) make entity dictionary embedding by one-hot encoding way
  def make_entity_embedding(self, x_train):
    X_train_vect = []
    entity_zeros = [0] * 7

    for sentence in x_train:
      sentence_idx = []
      for word in sentence:
        word_idx = []
        # check movie, actor, director, genre, country
        for one_dict in self.domain_dict_list:    
          if word in one_dict:
            word_idx.append(1)
          else:
            word_idx.append(0)
            
        # year
        year_regex = re.compile('\d+년|[12]\d{3}')
        if len(year_regex.findall(word)) == 1:
          word_idx.append(1)
        else:
          word_idx.append(0)
          
        # month
        month_regex = re.compile('\d+월')
        if len(month_regex.findall(word)) == 1:
          word_idx.append(1)
        else:
          word_idx.append(0)

        sentence_idx.append(word_idx)
        
      while (len(sentence_idx) < self.parameter['max_sent_len']):
        sentence_idx.append(entity_zeros)
      X_train_vect.append(sentence_idx)

    return np.array(X_train_vect)
  
  # (4) make syllable embedding
  def make_syl_embedding(self, x_train):
    syl_hidden_size = 8
    X_train = []
    X_train_vect = []
    max_idx = len(self.syl2idx)
    one_hot_matrix = np.eye(max_idx)
    one_hot_zeros = np.zeros([max_idx])
    word_zeros = np.zeros([syl_hidden_size*2])
    
    # build syllable model and load pretrained variables
    self.model.build_syllable_model(self.syl2idx)
    
    # syllable to idx
    for sent in x_train:
      tmp = []
      for word in sent:
        tmp_2 = []
        for syl in word:
          try:
            tmp_2.append(self.syl2idx[syl])
          except:
            tmp_2.append(max_idx)
        tmp.append(tmp_2)  
      X_train.append(tmp)
      
    # word with syllable indexes -> syllable vectors  
    for sent in X_train:
      sent = tf.keras.preprocessing.sequence.pad_sequences(sent, maxlen=self.parameter['max_word_len'], padding='post', value=max_idx)
      sent_vect = []
      for word in sent:
        one_hot_word = []
        x_input = []
        for syl in word:
          try:
            one_hot_word.append(one_hot_matrix[syl])
          except:
            one_hot_word.append(one_hot_zeros)
        x_input.append(one_hot_word)
        # run model and return syl_states vectors
        syl_states_ = self.model.run_syllable_model(x_input)
        
        word_vect = np.concatenate([syl_states_[1][0][0], syl_states_[0][0][0]])
        sent_vect.append(word_vect)

      while(len(sent_vect) < self.parameter['max_sent_len']):
        sent_vect.append(word_zeros)

      X_train_vect.append(sent_vect)
    
    self.model.close_session()

    return np.array(X_train_vect)
  
  # make concatenated embedding
  def make_concat_embedding(self, x_train, x_train_pos):
    wiki_embedding = self.make_word2vec_embedding(x_train, 'wiki')
    movie_embedding = self.make_word2vec_embedding(x_train, 'movie')
    pos_embedding = self.make_pos_embedding(x_train_pos)
    entity_embedding = self.make_entity_embedding(x_train)
    syl_embedding = self.make_syl_embedding(x_train)

    concat_embedding = np.concatenate([wiki_embedding, movie_embedding, pos_embedding, entity_embedding, syl_embedding], axis=-1)

    return concat_embedding