# -*- coding: utf-8 -*-

import pandas as pd

def load_data(file_path):
  data = pd.read_csv(file_path, names=['sentence', 'intent'], sep='\t', header=0)
  data = data['sentence']

  return data