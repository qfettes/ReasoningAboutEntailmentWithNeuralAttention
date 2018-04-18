import numpy
import sys
import time
import torch
import torch.nn as NN
import torch.optim as OPT
import torch.nn.functional as F
from torch.autograd import Variable


# =============================================================================
# Model 1: Train one LSTM network to process both the premise and the hypothesis.
#
#  This model uses the same LSTM network to create vector representations
#  for the premise and the hypothesis, then uses the concatenation of the 
#  two vector representations as the representation of the sentence pair
#  to be used as input to a softmax layer.
#
#  lstm_size: the size of the LSTM cell.
#  hidden_size: the size of the fully connected layers.
#  drop_rate: Dropout rate.
#  beta: the L2 regularizer parameter for the fully connected layers.
#  rep_1: the matrix of word embeddings for the premise sentence.
#  len_1: the true length of the premise sentence.
#  mask_1: binary vector specifying true words (1) and dummy words used for padding (0).
#  rep_2: the matrix of word embeddings for the hypothesis sentence.
#  len_2: the true length of the hypothesis sentence.
#
class Model_1(NN.Module):
    def __init__(self, use_gpu, lstm_size, hidden_size, drop_out, beta, embedding, class_num):
        super(Model_1, self).__init__()

        numpy.random.seed(2)
        torch.manual_seed(2)

        # Set tensor type when using GPU
        if use_gpu:
            self.float_type = torch.cuda.FloatTensor
            self.long_type = torch.cuda.LongTensor
            torch.cuda.manual_seed_all(2)
        # Set tensor type when using CPU
        else:
            self.float_type = torch.FloatTensor
            self.long_type = torch.LongTensor

        # Define parameters for model
        self.lstm_size = lstm_size
        self.hidden_size = hidden_size
        self.embedding = embedding
        input_size = self.embedding.weight.size()[1]
        self.drop_out = drop_out

        # The LSTM: input size, lstm size, num of layer
        self.lstm = NN.LSTM(input_size, lstm_size, 1)

        # The fully connectedy layers
        self.linear1 = NN.Linear(lstm_size + lstm_size, hidden_size)
        self.linear2 = NN.Linear(hidden_size, hidden_size)
        self.linear3 = NN.Linear(hidden_size, hidden_size)

        # The fully connected layer for softmax
        self.linear4 = NN.Linear(hidden_size, class_num)


    # init hidden states and cell states of LSTM
    def init_hidden(self, batch_size):
        return (Variable(torch.zeros(1, batch_size, self.lstm_size), 
                         requires_grad=False).type(self.float_type),
                Variable(torch.zeros(1, batch_size, self.lstm_size), 
                         requires_grad=False).type(self.float_type))


    # forward propagation
    def forward(self, rep1, len1, mask1, rep2, len2):
        rep = torch.cat((rep1, rep2), 0)
        length = torch.cat((len1, len2), 0)

        # Representation for input sentences
        batch_size = rep1.size()[0]
        sents = self.embedding(rep)

        # (sequence length * batch size * feature size)
        sents = sents.transpose(1, 0)

        # Initialize hidden states and cell states
        hidden = self.init_hidden(batch_size + batch_size)

        # Ouput of LSTM: sequence (length x mini batch x lstm size)
        lstm_outs, hidden = self.lstm(sents, hidden)

        # (batch size * sequence length * feature size)
        lstm_outs = lstm_outs.transpose(1, 0)

        # Get the valid output by the real length of the input sentences
        length = (length-1).view(-1, 1, 1).expand(lstm_outs.size(0), 1, lstm_outs.size(2))
        lstm_out = torch.gather(lstm_outs, 1, length)
        lstm_out = lstm_out.view(lstm_out.size(0), -1)

        # split representation to premise and hyphothesis representation
        (lstm_1_out, lstm_2_out) = torch.split(lstm_out, batch_size)

        # Concatenate premise and hypothesis representations
        lstm_1_out = F.dropout(lstm_1_out, p=self.drop_out)
        lstm_2_out = F.dropout(lstm_2_out, p=self.drop_out)
        lstm_out = torch.cat((lstm_1_out, lstm_2_out), 1)

        # Output of fully connected layers 
        fc_out = F.dropout(F.tanh(self.linear1(lstm_out)), p=self.drop_out)
        fc_out = F.dropout(F.tanh(self.linear2(fc_out)), p=self.drop_out)
        fc_out = F.dropout(F.tanh(self.linear3(fc_out)), p=self.drop_out)

        # Output of Softmax
        fc_out = self.linear4(fc_out)

        print(fc_out.shape)
        exit()
        return F.log_softmax(fc_out, dim=1)
 

# =============================================================================
# Model 2: Conditional encoding.
#
#  This model used two LSTM networks, one for the premise, one for the
#  hypothesis. The initial cell state of the hypothesis LSTM is set to be 
#  the last cell state of the premise LSTM. The last output of the
#  hypothesis LSTM is used as the representation of the sentence pair.
#
#  lstm_size: the size of the LSTM cell.
#  hidden_size: the size of the fully connected layers.
#  drop_rate: Dropout rate.
#  beta: the L2 regularizer parameter for the fully connected layers.
#  rep_1: the matrix of word embeddings for the premise sentence.
#  len_1: the true length of the premise sentence.
#  mask_1: binary vector specifying true words (1) and dummy words used for padding (0).
#  rep_2: the matrix of word embeddings for the hypothesis sentence.
#  len_2: the true length of the hypothesis sentence.
#
class Model_2(NN.Module):
    def __init__(self, use_gpu, lstm_size, hidden_size, drop_out, beta, embedding, class_num):
        super(Model_1, self).__init__()

        numpy.random.seed(2)
        torch.manual_seed(2)

        # Set tensor type when using GPU
        if use_gpu:
            self.float_type = torch.cuda.FloatTensor
            self.long_type = torch.cuda.LongTensor
            torch.cuda.manual_seed_all(2)
        # Set tensor type when using CPU
        else:
            self.float_type = torch.FloatTensor
            self.long_type = torch.LongTensor

        # Define parameters for model
        self.lstm_size = lstm_size
        self.hidden_size = hidden_size
        self.embedding = embedding
        feature_size = self.embedding.weight.size()[1]
        self.drop_out = drop_out

        # The LSTMs:
        # lstm1: premise; lstm2: hypothesis
        self.lstm1 = NN.LSTMCell(feature_size, lstm_size)
        self.lstm2 = NN.LSTM(feature_size, lstm_size, 1)

        # The fully connectedy layers
        self.linear1 = NN.Linear(lstm_size, hidden_size)
        self.linear2 = NN.Linear(hidden_size, hidden_size)
        self.linear3 = NN.Linear(hidden_size, hidden_size)

        # The fully connectedy layer for softmax
        self.linear4 = NN.Linear(hidden_size, class_num)


    # Initialize the hidden states and cell states of LSTM
    def init_hidden(self, batch_size):
        return (Variable(torch.zeros(1, batch_size, self.lstm_size), 
                         requires_grad=False).type(self.float_type),
                Variable(torch.zeros(1, batch_size, self.lstm_size), 
                         requires_grad=False).type(self.float_type))


    # Forward propagation
    def forward(self, rep1, len1, mask1, rep2, len2):
    # ----------------- YOUR CODE HERE ----------------------
        #rep = torch.cat((rep1, rep2), 0)
        #length = torch.cat((len1, len2), 0)

        # Representation for input sentences
        batch_size = rep1.size()[0]
        sents_premise = self.embedding(rep1)
        sents_hypothesis = self.embedding(rep2)

        # (sequence length * batch size * feature size)
        sents_premise = sents_premise.transpose(1, 0)
        sents_hypothesis = sents_hypothesis.transpose(1, 0)

        # Initialize hidden states and cell states
        hidden = self.init_hidden(batch_size)
        hidden_hypothesis = self.init_hidden(batch_size)

        # Ouput of LSTM: sequence (length x mini batch x lstm size)
        outp = []
        for i, inp in enumerate(sents_premise.chunk(sents_premise.size(0), dim=0)):
            print(sents_premise)
            exit()
            hidden = self.lstm1(inp, hidden)
            outp += [hidden[1]]
            
        outp = torch.stack(outp).squeeze(dim=1).transpose(1, 0)
        len1 = (len1-1).view(-1, 1, 1).expand(outp.size(0), 1, outp.size(2))
        out = torch.gather(outp, 1, len1).transpose(1, 0)
        
        lstm_outs, hidden_hypothesis = self.lstm2(sents_hypothesis, (hidden_hypothesis[0], out))

        # (batch size * sequence length * feature size)
        lstm_out = hidden_hypothesis[1].squeeze(0)
        # Concatenate premise and hypothesis representations
        lstm_out = F.dropout(lstm_out, p=self.drop_out)

        # Output of fully connected layers 
        fc_out = F.dropout(F.tanh(self.linear1(lstm_out)), p=self.drop_out)
        fc_out = F.dropout(F.tanh(self.linear2(fc_out)), p=self.drop_out)
        fc_out = F.dropout(F.tanh(self.linear3(fc_out)), p=self.drop_out)

        # Output of Softmax
        fc_out = self.linear4(fc_out)

        return F.log_softmax(fc_out)
    # -------------------------------------------------------
 

# =============================================================================
# Model 3: Use attention for last LSTM output of hypothesis.
#
#  This model use an attention mechanism, where the attention weights
#  are computed between the last output of the hypothesis LSTM and all 
#  the outputs of the premise LTSM.  
#
#  lstm_size: the size of the LSTM cell.
#  hidden_size: the size of the fully connected layers.
#  drop_rate: Dropout rate.
#  beta: the L2 regularizer parameter for the fully connected layers.
#  rep_1: the matrix of word embeddings for the premise sentence.
#  len_1: the true length of the premise sentence.
#  mask_1: binary vector specifying true words (1) and dummy words used for padding (0).
#  rep_2: the matrix of word embeddings for the hypothesis sentence.
#  len_2: the true length of the hypothesis sentence.
#
class Model_3(NN.Module):
    def __init__(self, use_gpu, lstm_size, hidden_size, drop_out, beta, embedding, class_num):
        super(Model_3, self).__init__()

        numpy.random.seed(2)
        torch.manual_seed(2)

        # Set tensor type when using GPU
        if use_gpu:
            self.float_type = torch.cuda.FloatTensor
            self.long_type = torch.cuda.LongTensor
            torch.cuda.manual_seed_all(2)
        # Set tensor type when using CPU
        else:
            self.float_type = torch.FloatTensor
            self.long_type = torch.LongTensor

        # Define parameters for model
        self.lstm_size = lstm_size
        self.hidden_size = hidden_size
        self.embedding = embedding
        input_size = self.embedding.weight.size()[1]
        self.drop_out = drop_out

        # The LSTMs: lstm1 - premise; lstm2 - hypothesis
        self.lstm1 = NN.LSTMCell(input_size, lstm_size)
        self.lstm2 = NN.LSTM(input_size, lstm_size, 1)

        # The fully connectedy layers
        self.linear1 = NN.Linear(lstm_size, hidden_size)

        # The fully connectedy layer for softmax
        self.linear2 = NN.Linear(hidden_size, class_num)

        # transformation of the states
        u_min = -0.5
        u_max =  0.5

        self.Wy = NN.Parameter(torch.Tensor(lstm_size, lstm_size).uniform_(u_min, u_max))
        self.Wh = NN.Parameter(torch.Tensor(lstm_size, lstm_size).uniform_(u_min, u_max))
        self.Wp = NN.Parameter(torch.Tensor(lstm_size, lstm_size).uniform_(u_min, u_max))
        self.Wx = NN.Parameter(torch.Tensor(lstm_size, lstm_size).uniform_(u_min, u_max))
        self.aW = NN.Parameter(torch.Tensor(1, lstm_size).uniform_(u_min, u_max))


    # Initialize hidden states and cell states of LSTM
    def init_hidden(self, batch_size):
        return (Variable(torch.zeros(1, batch_size, self.lstm_size), 
                         requires_grad=False).type(self.float_type),
                Variable(torch.zeros(1, batch_size, self.lstm_size), 
                         requires_grad=False).type(self.float_type))


    # Forward propagation
    def forward(self, rep1, len1, mask1, rep2, len2):

        # Compute context vectors using attention.
        def context_vector(h_t):
            WhH = torch.matmul(h_t, self.Wh)

            # Use mask to ignore the outputs of the padding part in premise
            shape = WhH.size()
            WhH = WhH.view(shape[0], 1, shape[1])
            WhH = WhH.expand(shape[0], max_seq_len, shape[1])

            M1 = mask1.type(self.float_type)
            shape = M1.size()
            M = M1.view(shape[0], shape[1], 1).type(self.float_type)
            M = M.expand(shape[0], shape[1], self.lstm_size)

            WhH = WhH * M
            M = torch.tanh(WyY + WhH)
            aW = self.aW.view(1, 1, -1)
            aW = aW.expand(batch_size, max_seq_len, aW.size()[2])

            # Compute batch dot: the first step of a softmax
            batch_dot = M * aW
            batch_dot = torch.sum(batch_dot, 2)

            # Avoid overflow
            max_by_column, _ = torch.max(batch_dot, 1)
            max_by_column = max_by_column.view(-1, 1)
            max_by_column = max_by_column.expand(max_by_column.size()[0], max_seq_len)

            batch_dot = torch.exp(batch_dot - max_by_column) * M1

            # Partition function and attention: 
            # the second step of a softmax, use mask to ignore the padding
            partition = torch.sum(batch_dot, 1)
            partition = partition.view(-1, 1)
            partition = partition.expand(partition.size()[0], max_seq_len)
            attention = batch_dot / partition

            # compute context vector
            shape = attention.size()
            attention = attention.view(shape[0], shape[1], 1)
            attention = attention.expand(shape[0], shape[1], self.lstm_size)

            cv_t = outputs_1 * attention
            cv_t = torch.sum(cv_t, 1)

            return cv_t

        # ################# Forward Propagation code ###################

        # Set batch size
        batch_size = rep1.size()[0]

        # Representation of input sentences
        sent1 = self.embedding(rep1)
        sent2 = self.embedding(rep2)

        # Transform sentences representations to:
        # (sequence length * batch size * feqture size)
        sent1 = sent1.transpose(1, 0)
        sent2 = sent2.transpose(1, 0)

        # ----------------- YOUR CODE HERE ----------------------
        # Run the two LSTM's, compute the context vectors,
        # compute the final representation of the sentence pair,
        # and run it through the fully connected layer, then 
        # through the softmax layer.

























        # -------------------------------------------------------
 

###############################################################
# Recurrent neural network class
#
class RNNNet(object):
    def __init__(self, mode):
        self.mode = mode

        # Set tensor type when using GPU
        if torch.cuda.is_available():
            self.use_gpu = True
            self.float_type = torch.cuda.FloatTensor
            self.long_type = torch.cuda.LongTensor
        # Set tensor type when using CPU
        else:
            self.use_gpu = False
            self.float_type = torch.FloatTensor
            self.long_type = torch.LongTensor

    # Get a batch of data from given data set.
    def get_batch(self, data_set, s, e):
        sent_1 = data_set[0]
        len_1  = data_set[1]
        sent_2 = data_set[2]
        len_2  = data_set[3]
        label  = data_set[4]
        return sent_1[s:e], len_1[s:e], sent_2[s:e], len_2[s:e], label[s:e]

    # Create mask for premise sentences.
    def create_mask(self, data_set, max_length):
        length = data_set[1]
        masks = []
        for one in length:
            mask = list(numpy.ones(one))
            mask.extend(list(numpy.zeros(max_length - one)))
            masks.append(mask)
        masks = numpy.asarray(masks, dtype=numpy.float32)    
        return masks

    # Evaluate the trained model on test set
    def evaluate_model(self, pred_Y, Y):
        _, idx = torch.max(pred_Y, dim=1)

        # move tensor from GPU to CPU when using GPU
        if self.use_gpu:
            idx = idx.cpu()
            Y = Y.cpu()

        idx = idx.data.numpy()
        Y = Y.data.numpy()
        accuracy = numpy.sum(idx == Y)
        return accuracy

    # Train and evaluate SNLI models
    def train_and_evaluate(self, FLAGS, embedding, train_set, dev_set, test_set):
        class_num     = 3
        num_epochs    = FLAGS.num_epochs
        batch_size    = FLAGS.batch_size
        learning_rate = FLAGS.learning_rate

        beta        = FLAGS.beta
        drop_rate   = FLAGS.dropout_rate
        lstm_size   = FLAGS.lstm_size
        hidden_size = FLAGS.hidden_size

        # Word embeding
        vectors = embedding.vectors

        # Max length of input sequence
        max_seq_len = train_set[0].shape[1]

        # Create mask for first sentence
        train_mask = self.create_mask(train_set, max_seq_len)
        dev_mask = self.create_mask(dev_set, max_seq_len)
        test_mask = self.create_mask(test_set, max_seq_len)

        # Train, validate and test set size
        train_size = train_set[0].shape[0]
        dev_size = dev_set[0].shape[0]
        test_size = test_set[0].shape[0]

        # Initialize embedding matrix
        embedding = NN.Embedding(vectors.shape[0], vectors.shape[1], padding_idx=0)
        embedding.weight = NN.Parameter(torch.from_numpy(vectors))
        embedding.weight.requires_grad = False

        # uncomment the below three lines to force the code to use CPU
        # self.use_gpu = False
        # self.float_type = torch.FloatTensor
        # self.long_type = torch.LongTensor

        # Define models
        model = eval("Model_" + str(self.mode))(
                    self.use_gpu, lstm_size, hidden_size, drop_rate, beta, embedding, class_num
                )
    
        # If GPU is available, then run experiments on GPU
        if self.use_gpu:
            model.cuda()

        # ======================================================================
        # define optimizer
        #
        optimizer = OPT.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                             lr=learning_rate)

        # ======================================================================
        # accuracy calculation
        #
        accuracy = 0

        for i in range(num_epochs):
            # put model to training mode
            model.train()

            print(20 * '*', 'epoch', i+1, 20 * '*')
            start_time = time.time()
            s = 0
            while s < train_size:
                model.train()
                e = min(s + batch_size, train_size)

                batch_1v, batch_1l, batch_2v, batch_2l, batch_label = \
                    self.get_batch(train_set, s, e)
                mask = train_mask[s:e]
            
                rep1 = Variable(torch.from_numpy(batch_1v),
                                requires_grad=False).type(self.long_type)
                len1 = Variable(torch.from_numpy(batch_1l),
                                requires_grad=False).type(self.long_type)
                rep2 = Variable(torch.from_numpy(batch_2v),
                                requires_grad=False).type(self.long_type)
                len2 = Variable(torch.from_numpy(batch_2l),
                                requires_grad=False).type(self.long_type)
                mask = Variable(torch.from_numpy(mask),
                                requires_grad=False).type(self.long_type)
                label = Variable(torch.from_numpy(batch_label),
                                 requires_grad=False).type(self.long_type)

                # Forward pass: predict labels
                pred_label = model(rep1, len1, mask, rep2, len2)

                # Loss function: compute negative log likelyhood
                loss = F.nll_loss(pred_label, label)

                # Zero gradients, perform a backward pass, and update the weights.
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                s = e

            end_time = time.time()
            print ('the training took: %d(s)' % (end_time - start_time))

            # Put model in evaluation mode
            model.eval()

            # Evaluate the trained model on validation set
            s = 0
            total_correct = 0
            while s < dev_size:
                e = min(s + batch_size, dev_size)
                batch_1v, batch_1l, batch_2v, batch_2l, batch_label = \
                    self.get_batch(dev_set, s, e)
                mask = dev_mask[s:e]

                rep1 = Variable(torch.from_numpy(batch_1v),
                                requires_grad=False).type(self.long_type)
                len1 = Variable(torch.from_numpy(batch_1l),
                                requires_grad=False).type(self.long_type)
                rep2 = Variable(torch.from_numpy(batch_2v),
                                requires_grad=False).type(self.long_type)
                len2 = Variable(torch.from_numpy(batch_2l),
                                requires_grad=False).type(self.long_type)
                mask = Variable(torch.from_numpy(mask),
                                requires_grad=False).type(self.long_type)
                label = Variable(torch.from_numpy(batch_label),
                                 requires_grad=False).type(self.long_type)

                # Forward pass: predict labels
                pred_label = model(rep1, len1, mask, rep2, len2)

                total_correct += self.evaluate_model(pred_label, label)

                s = e

            print ('accuracy of the trained model on validation set %f' % 
                    (total_correct / dev_size))
            print ()

            # evaluate the trained model on test set
            s = 0
            total_correct = 0
            while s < test_size:
                e = min(s + batch_size, test_size)
                batch_1v, batch_1l, batch_2v, batch_2l, batch_label = \
                    self.get_batch(test_set, s, e)
                mask = test_mask[s:e]

                rep1 = Variable(torch.from_numpy(batch_1v),
                                requires_grad=False).type(self.long_type)
                len1 = Variable(torch.from_numpy(batch_1l),
                                requires_grad=False).type(self.long_type)
                rep2 = Variable(torch.from_numpy(batch_2v),
                                requires_grad=False).type(self.long_type)
                len2 = Variable(torch.from_numpy(batch_2l),
                                requires_grad=False).type(self.long_type)
                mask = Variable(torch.from_numpy(mask),
                                requires_grad=False).type(self.long_type)
                label = Variable(torch.from_numpy(batch_label),
                                 requires_grad=False).type(self.long_type)

                # Forward pass: predict labels
                pred_label = model(rep1, len1, mask, rep2, len2)

                total_correct += self.evaluate_model(pred_label, label)

                s = e

        return total_correct / test_size


