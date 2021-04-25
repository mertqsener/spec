'''
Expected run times on a GTX 1080 GPU:
MNIST: 1 hr
Reuters: 2.5 hrs
cc: 15 min
'''

import sys, os
# add directories in src/ to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))
import csv
import argparse
from collections import defaultdict
import numpy as np
from core.data import get_data
from spectralnet import run_net

params = {
    'dset': 'new_dataset',
    'val_set_fraction': 0.1,
    'siam_batch_size': 128,
    'n_clusters': 10,
    'affinity': 'knn',
    'n_nbrs': 3,
    'scale_nbrs': 2,
    'siam_k': 2,
    'siam_ne': 10,
    'spec_ne': 10,
    'siam_lr': 1e-3,
    'spec_lr': 1e-3,
    'siam_patience': 10,
    'spec_patience': 20,
    'siam_drop': 0.1,
    'spec_drop': 0.1,
    'batch_size': 1024,
    'siam_reg': None,
    'spec_reg': None,
    'siam_n': None,
    'siamese_tot_pairs': 600000,
    'arch': [
        {'type': 'relu', 'size': 1024},
        {'type': 'relu', 'size': .512},
        {'type': 'relu', 'size': 10},
        ],
    'use_approx': False,
    }
def load_new_dataset_data():  
 rows = []
 
 # load dataset
 with open('./in_X.csv', 'r') as csvfile:
    # creating a csv reader object
    x = csv.reader(csvfile)
      
    # extracting field names through first row
    fields = next(x)
  
    # extracting each data row one by one
    for row in x:
        rows.append(row)
 with open('./true_labs.csv', 'r') as csvfile:
    # creating a csv reader object
    y = csv.reader(csvfile)
      
    # extracting field names through first row
    fields = next(y)
  
    # extracting each data row one by one
    for row in y:
        rows.append(row)
  
     
  # make train and test splits
    n_train = int(0.9 * len(x))
    x_train, x_test = x[:n_train], x[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

 return x_train, x_test, y_train, y_test

x_train, x_test, y_train, y_test = load_new_dataset_data()
new_dataset_data = (x_train, x_test, y_train, y_test)

# preprocess dataset
data = get_data(params, new_dataset_data)

# run spectral net
x_spectralnet, y_spectralnet = run_net(data, params)
