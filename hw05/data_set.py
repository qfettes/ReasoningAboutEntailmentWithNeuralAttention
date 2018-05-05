import json
import numpy
import pickle
import re
import sys
from nltk import tokenize
from struct import unpack


################################################################
# Read word embedding from binary file
#
class WordEmbedding(object):
    def __init__(self, input_file, vocab):
        self.word_to_id = {} # word to id map
        self.id_to_word = {} # id to word map
        self.vectors = self.read_embedding(input_file, vocab)

    # read words representation from given file
    def read_embedding(self, input_file, vocabulary):
        wid = 0
        em_list = []

        with open(input_file, 'rb') as f:
            cols = f.readline().strip().split()  # read first line
            vocab_size = int(cols[0].decode())   # get vocabulary size
            vector_size = int(cols[1].decode())  # get word vector size

            # add embedding for the padding word
            em_list.append(numpy.zeros([1, vector_size]))
            wid += 1

            # add embedding for out of vocabulary word
            self.word_to_id['<unk>'] = wid
            self.id_to_word[wid] = '<unk>'
            em_list.append(numpy.zeros([1, vector_size]))
            wid += 1

            # set read format: get vector for one word in one read operation
            fmt = str(vector_size) + 'f'

            for i in range(0, vocab_size, 1):
                # init one word with empty string
                vocab = b''

                # read char from the line till ' '
                ch = b''
                while ch != b' ':
                    vocab += ch
                    ch = f.read(1)
            
                # convert word from binary to string
                vocab = vocab.decode()

                # read one word vector
                word_vector = list(unpack(fmt, f.read(4 * vector_size))),
                one_vec = numpy.asarray(word_vector, dtype=numpy.float32)

                # If your embedding file has '\n' at the end of each line, 
                # uncomment the below line. 
                # If your embedding file has no '\n' at the end of each line,
                # comment the below line
                #f.read(1)

                if vocab not in vocabulary:
                    if vocab == 'unk':
                        em_list[1] = one_vec
                    continue

                # stored the word, word id and word representation
                self.word_to_id[vocab] = wid
                self.id_to_word[wid] = vocab
                em_list.append(one_vec)

                # increase word id
                wid += 1

        vectors = numpy.asarray(em_list, dtype=numpy.float32)
        vectors = vectors.reshape(vectors.shape[0], vectors.shape[2])
        return vectors


################################################################
# Read sentence pairs from SNLI data set
#
class SNLI(object):
    def __init__(self, embedding, snli_path):
        cols = snli_path.split('/')
        train_file = snli_path + '/' + cols[-1] + '_train.jsonl'
        dev_file = snli_path + '/' + cols[-1] + '_dev.jsonl'
        test_file = snli_path + '/' + cols[-1] + '_test.jsonl'

        if not embedding:
            self.vocab = set()
            self.collect_vocab([train_file, dev_file, test_file])
        else:
            self.word_to_id = embedding.word_to_id
            self.id_to_word = embedding.id_to_word
            self.vectors = embedding.vectors
            self.max_sent_len = 0
            self.label_dict = {'entailment' : 0, 
                               'neutral' : 1, 
                               'contradiction' : 2}

            self.train_set = self.load_data(train_file)
            self.dev_set = self.load_data(dev_file)
            self.test_set = self.load_data(test_file)

    # tokenize the given text
    def tokenize_text(self, text):
        text = text.replace('\\', '')
        text = re.sub(r'\.+', '.', text)

        # split text into sentences
        sents = tokenize.sent_tokenize(text)

        for sent in sents:
            # split sent into words
            tokens = tokenize.word_tokenize(sent)

            # ignore empty sentences
            if not tokens: 
                continue

            # create an iterator for tokenized words
            for token in tokens:
                ntokens = token.split('-')
                if len(ntokens) == 1:
                    yield token
                else:
                    for one in ntokens:
                        yield one

    # collect vocabulary of the SNLI set
    def collect_vocab(self, file_list):
        for one_file in file_list:
            for line in open(one_file, 'r'):
                one_dict = json.loads(line)

                # get word list for sentence 1
                for word in self.tokenize_text(one_dict['sentence1']):
                    self.vocab.add(word)
 
                # get word list for sentence 2
                for word in self.tokenize_text(one_dict['sentence2']):
                    self.vocab.add(word)

    # sentence pairs and their labels
    def load_data(self, input_file):
        sent1_list = []
        sent2_list = []
        label_list = []

        for line in open(input_file, 'r'):
            one_dict = json.loads(line)

            # read label
            label = one_dict['gold_label']
            if label == '-':
                continue
            label = self.label_dict[label]

            # get word list for sentence 1
            sentence1 = []
            for x in self.tokenize_text(one_dict['sentence1']):
                if x in self.word_to_id:
                    sentence1.append(self.word_to_id[x])
                else:
                    sentence1.append(1)
            self.max_sent_len = max(self.max_sent_len, len(sentence1))
 
            # get word list for sentence 2
            sentence2 = []
            for x in self.tokenize_text(one_dict['sentence2']):
                if x in self.word_to_id:
                    sentence2.append(self.word_to_id[x])
                else:
                    sentence2.append(1)
            self.max_sent_len = max(self.max_sent_len, len(sentence2))

            sent1_list.append(sentence1)
            sent2_list.append(sentence2)
            label_list.append(label)

        return [sent1_list, sent2_list, label_list]

    def list_to_array(self, sent_list, max_len):
        selist = []
        length = []
        for one in sent_list:
            length.append(len(one))
            if len(one) < max_len:
                one.extend(list(numpy.zeros(max_len - len(one), 
                                            dtype=numpy.int32)))
            selist.append(one)

        selist = numpy.asarray(selist, dtype=numpy.int32)
        length = numpy.asarray(length, dtype=numpy.int32)

        return selist, length

    def create_padding(self, data_set):
        sent_1v, sent_1l = self.list_to_array(data_set[0], self.max_sent_len)
        sent_2v, sent_2l = self.list_to_array(data_set[1], self.max_sent_len)
        data = [sent_1v, sent_1l, sent_2v, sent_2l, 
                numpy.asarray(data_set[2], dtype=numpy.int32)]
        return data

    def create_padding_set(self):
        train_set = self.create_padding(self.train_set)
        dev_set = self.create_padding(self.dev_set)
        test_set = self.create_padding(self.test_set)
        return train_set, dev_set, test_set

def main():
    # collect vocabulary of SNLI set
    snli = SNLI(None, sys.argv[1])

    # read word embedding
    embedding = WordEmbedding(sys.argv[2], snli.vocab)
    pickle.dump(embedding, open(sys.argv[3], 'wb'))

    # create SNLI data set
    snli = SNLI(embedding, sys.argv[1])
    train_set, dev_set, test_set = snli.create_padding_set()
    pickle.dump([train_set, dev_set, test_set], open(sys.argv[4], 'wb'))

if __name__ == '__main__':
    main()

