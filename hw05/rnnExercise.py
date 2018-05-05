# ==============================================================================
# DL6890 RNN Exercise

#  Instructions
#  ------------
# 
#  This file contains code that helps you get started on the RNN.
#  You will need to complete code in train_rnn.py
#
# ======================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy
import pickle
import sys

from data_set import WordEmbedding
from train_rnn import RNNNet


# Set parameters for RNN Exercise.
parser = argparse.ArgumentParser('RNN Exercise.')
parser.add_argument('--embedding_path', 
                    type=str, 
                    default='data/embedding.pkl', 
                    help='Path of the pretrained word embedding.')
parser.add_argument('--snli_data_dir', 
                    type=str, 
                    default='data/snli_padding.pkl', 
                    help='Directory to put the snli data.')
parser.add_argument('--log_dir', 
                    type=str, 
                    default='logs', 
                    help='Directory to put logging.')
parser.add_argument('--num_epochs',
                    type=int,
                    default=10, 
                    help='Number of epochs to run trainer.')
parser.add_argument('--batch_size', 
                    type=int,
                    default=512, 
                    help='Batch size. Must divide evenly into the dataset sizes.')
parser.add_argument('--learning_rate', 
                    type=float,
                    default=0.001,
                    help='Initial learning rate.')
parser.add_argument('--beta',
                    type=float,
                    default=0.001,
                    help='Decay rate of L2 regulization.')
parser.add_argument('--dropout_rate',
                    type=float,
                    default=0.1,
                    help='Dropout rate.')
parser.add_argument('--lstm_size',
                    type=int,
                    default=100,
                    help='Size of lstm cell.')
parser.add_argument('--hidden_size',
                    type=int,
                    default=200,
                    help='Size of hidden layer of FFN.')

FLAGS = None
FLAGS, unparsed = parser.parse_known_args()
mode = int(sys.argv[1])

# ======================================================================
#  STEP 0: Load pre-trained word embeddings and the SNLI data set
#

embedding = pickle.load(open(FLAGS.embedding_path, 'rb'))

snli = pickle.load(open(FLAGS.snli_data_dir, 'rb'))
train_set = snli[0]
dev_set   = snli[1]
test_set  = snli[2]

# ====================================================================
# Use a smaller portion of training examples (e.g. ratio = 0.1) 
# for debuging purposes.
# Set ratio = 1 for training with all training examples.

ratio = 1

train_size = train_set[0].shape[0]
idx = list(range(train_size))
idx = numpy.asarray(idx, dtype=numpy.int32)

# Shuffle the train set.
for i in range(7):
  numpy.random.seed(i)
  numpy.random.shuffle(idx)

# Get a certain ratio of the training set.
idx = idx[0:int(idx.shape[0] * ratio)]
sent1 = train_set[0][idx]
leng1 = train_set[1][idx]
sent2 = train_set[2][idx]
leng2 = train_set[3][idx]
label = train_set[4][idx]

train_set = [sent1, leng1, sent2, leng2, label]

# ======================================================================
#  STEP 1: Train the first model.
#  This model uses the same LSTM network to create vector representations
#  for the premise and the hypothesis, then uses the concatenation of the 
#  two vector representations as the representation of the sentence pair
#  to be used as input to a softmax layer.
#
#  Accuracy: 76.9%
#

if mode == 1:
  rnn = RNNNet(1)
  accuracy = rnn.train_and_evaluate(FLAGS, embedding, train_set, dev_set, test_set)

  #=======================================================================
  # output accuracy
  #
  print(20 * '*' + 'model 1' + 20 * '*')
  print('accuracy is %f' % (accuracy))
  print()


# ======================================================================
#  STEP 2: Train the second model.
#  This model used two LSTM networks, one for the premise, one for the
#  hypothesis. The initial cell state of the hypothesis LSTM is set to be 
#  the last cell state of the premise LSTM. The last output of the
#  hypothesis LSTM is used as the representation of the sentence pair.
#  
#  Accuracy: 77.6%
#

if mode == 2:
  rnn = RNNNet(2)
  accuracy = rnn.train_and_evaluate(FLAGS, embedding, train_set, dev_set, test_set)

  # ======================================================================
  # output accuracy
  #
  print(20 * '*' + 'model 2' + 20 * '*')
  print('accuracy is %f' % (accuracy))
  print()


# ======================================================================
#  STEP 3: Train the third model.
#  This model use an attention mechanism, where the attention weights
#  are computed between the last output of the hypothesis LSTM and all 
#  the outputs of the premise LTSM.  
#
#  Accuracy: 79.2%
#

if mode == 3:
  rnn = RNNNet(3)
  accuracy = rnn.train_and_evaluate(FLAGS, embedding, train_set, dev_set, test_set)

  # ======================================================================
  # output accuracy
  #
  print(20 * '*' + 'model 3' + 20 * '*')
  print('accuracy is %f' % (accuracy))
  print()


