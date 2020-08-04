'''Author: Rongjie Wang'''

from __future__ import print_function
from keras.callbacks import LambdaCallback, Callback
from keras.layers import Dense, Bidirectional, Average, average, Input
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import LSTM
from keras.layers import Conv1D, MaxPooling1D, AveragePooling1D
from keras.optimizers import RMSprop
from keras.callbacks import TensorBoard
from keras.utils.data_utils import get_file
from keras.models import model_from_json
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.callbacks import CSVLogger
from keras.optimizers import RMSprop
from keras.models import Model
from keras.layers.normalization import BatchNormalization
import numpy as np
import random
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from attention import AttentionWithContext
import sys
import io
import datetime
import math
import keras
from keras.models import load_model
from keras import backend as K
from itertools import product
from Bio import SeqIO
from keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(monitor='val_loss', patience=2)
np.random.seed(1337)  # for reproducibility
train_path = '/home/cww/DeepZip/rongjiewang/DNN_dataset/train.fasta'
valid_path = '/home/cww/DeepZip/rongjiewang/DNN_dataset/valid.fasta'
#test_path = '/home/cww/DeepZip/rongjiewang/DNN_dataset/test.fasta'
test_path = "/home/cww/DeepZip/DNA/fish_xlt_final/test.fasta"
todaydate = str(datetime.date.today()) + 'CNN+LSTM'
chars = "ACGTN"
print('total chars:', len(chars))
print('chars:', chars)
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))
maxlen = 64
step = 1
batch_size = 128
epochs = 10
input_dim = len(chars)


def loss_fn(y_true, y_pred):
    return 1 / np.log(2) * K.categorical_crossentropy(y_true, y_pred)


def read_fasta(data_path):
    records = list(SeqIO.parse(data_path, "fasta"))
    text = ""
    for record in records:
        text += str(record.seq)
    return text


def read_data(text):

    for i in range(0, len(text) - maxlen, step):
        sentence = text[i: i + maxlen]
        next_char = text[i + maxlen]
        yield sentence, next_char


def vectorization(sentences, next_chars):
    x = np.zeros((maxlen, len(chars)), dtype=np.bool)
    y = np.zeros((len(chars)), dtype=np.bool)
    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            x[t, char_indices[char]] = 1
        y[char_indices[next_chars[i]]] = 1
    return x, y


def get_batch(stream):
    sentences = []
    next_chars = []
    for sentence, next_char in stream:
        sentences.append(sentence)
        next_chars.append(next_char)

        data_tuple = vectorization(sentences, next_chars)
        yield data_tuple
        sentences = []
        next_chars = []


def my_kernel_initializer(shape, dtype=None):
    x = np.zeros(shape, dtype=np.bool)
    for i, c in enumerate(product('ACGT', repeat=5)):
        kmer = c * 3
        for t, char in enumerate(kmer):
            x[t, char_indices[char], i] = 1
    return x


def loadModel():
    # model.load_weights('my_model_weights.h5')
    # json and create model
    json_file = open('./model/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights("./model/model.h5")
    print("Loaded model from disk")
    return model
def Model_CNN_BiLSTM_noCNN(alphabet_size): 

   # convs = []
   # filter_sizes = [6,7,8]
    model = Sequential()

    
    model.add(BatchNormalization(input_shape=( maxlen,alphabet_size)))
    #model.add(Dropout(0.5))
    #model.add(Embedding(alphabet_size, 16, input_length= 5))
    model.add(Bidirectional(LSTM(256, stateful=False, return_sequences=True)))
    #model.add(Bidirectional(LSTM(128, stateful=False, return_sequences=True),merge_mode='concat'))
    # model.add(Bidirectional(CuDNNLSTM(32, stateful=False, return_sequences=False)))
    
    
    model.add(AttentionWithContext())
    #model.add(Activation('relu'))
    #model.add(BatchNormalization())
    #model.add(Flatten())
    #model.add(Dropout(0.2))
    #model.add(Dropout(0.2))
    #
    #model.add(Dense(1024))
    model.add(Activation('relu'))
   # model.add(BatchNormalization()
   # model.add(Dense(input_dim))
    #model.add(Activation('softmax'))
    model.add(Dense(alphabet_size, activation='softmax'))
    
    return model

def Model_CNN_BiLSTM_noBi(alphabet_size):  #cnn_emerge+biLSTM+attention

   # convs = []
   # filter_sizes = [6,7,8]
    model = Sequential()
    model.add(Conv1D(filters=1024,
                     kernel_size=24,
                     trainable=True,
                     padding='valid',
                     activation='relu',
                     strides=1,input_shape=( maxlen,alphabet_size)))
    model.add(MaxPooling1D(pool_size=3))
    
    model.add(BatchNormalization())
    #model.add(Dropout(0.5))
    #model.add(Embedding(alphabet_size, 16, input_length= 5))
    model.add(LSTM(256, stateful=False, return_sequences=True))
    #model.add(Bidirectional(LSTM(128, stateful=False, return_sequences=True),merge_mode='concat'))
    # model.add(Bidirectional(CuDNNLSTM(32, stateful=False, return_sequences=False)))
    
    
    model.add(AttentionWithContext())
    #model.add(Activation('relu'))
    #model.add(BatchNormalization())
    #model.add(Flatten())
    #model.add(Dropout(0.2))
    #model.add(Dropout(0.2))
    #
    #model.add(Dense(1024))
    model.add(Activation('relu'))
   # model.add(BatchNormalization()
   # model.add(Dense(input_dim))
    #model.add(Activation('softmax'))
    model.add(Dense(alphabet_size, activation='softmax'))
    
    return model
def Model_CNN_BiLSTM_noatten(alphabet_size):  #cnn_emerge+biLSTM+attention

   # convs = []
   # filter_sizes = [6,7,8]
    model = Sequential()
    model.add(Conv1D(filters=1024,
                     kernel_size=24,
                     trainable=True,
                     padding='valid',
                     activation='relu',
                     strides=1,input_shape=( maxlen,alphabet_size)))
    model.add(MaxPooling1D(pool_size=3))
    
    model.add(BatchNormalization())
    #model.add(Dropout(0.5))
    #model.add(Embedding(alphabet_size, 16, input_length= 5))
    model.add(Bidirectional(LSTM(256, stateful=False, return_sequences=True)))
    #model.add(Bidirectional(LSTM(128, stateful=False, return_sequences=True),merge_mode='concat'))
    # model.add(Bidirectional(CuDNNLSTM(32, stateful=False, return_sequences=False)))
    
    
   # model.add(AttentionWithContext())
    #model.add(Activation('relu'))
    #model.add(BatchNormalization())
    model.add(Flatten())
    #model.add(Dropout(0.2))
    #model.add(Dropout(0.2))
    #
    #model.add(Dense(1024))
   # model.add(Activation('relu'))
   # model.add(BatchNormalization()
   # model.add(Dense(input_dim))
    #model.add(Activation('softmax'))
    model.add(Dense(alphabet_size, activation='softmax'))
    
    return model
def Model1(bs,time_steps, alphabet_size):  
    model = Sequential()
    model.add(Embedding(alphabet_size, 16, batch_input_shape=(bs, time_steps)))
    model.add(Bidirectional(LSTM(128, stateful=False, return_sequences=True)))
    # model.add(Bidirectional(CuDNNLSTM(32, stateful=False, return_sequences=False)))
    model.add(Dropout(0.2))
    model.add(AttentionWithContext())
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))

    model.add(BatchNormalization())
    model.add(Dense(alphabet_size, activation='softmax'))
    return model

def Model():  # cnn_emerge+biLSTM+attention

    print('Build model...')
    # convs = []
    # filter_sizes = [6,7,8]
    model = Sequential()
    model.add(Conv1D(filters=1024,kernel_size=24,trainable=True, padding='valid',activation='relu',strides=1,input_shape=( maxlen,5)))
    model.add(MaxPooling1D(pool_size=3))

    model.add(BatchNormalization())
   # model.add(Dropout(0.1))
    # model.add(Embedding(alphabet_size, 16, input_length= 5))
    model.add(Bidirectional(LSTM(256, stateful=False, return_sequences=True)))
    # model.add(Bidirectional(LSTM(128, stateful=False, return_sequences=True),merge_mode='concat'))
    # model.add(Bidirectional(CuDNNLSTM(32, stateful=False, return_sequences=False)))

    model.add(AttentionWithContext())
    # model.add(Activation('relu'))
    # model.add(BatchNormalization())
    # model.add(Flatten())
    # model.add(Dropout(0.2))
    # model.add(Dropout(0.2))
    #
    # model.add(Dense(1024))
    model.add(Activation('relu'))
    #model.add(BatchNormalization())
    model.add(Dense(input_dim))
    model.add(Activation('softmax'))
    #model.add(Dense(input_dim, activation='softmax'))
   

    return model



def saveModel(epoch):
    # serialize model to JSON
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    name = "model_" + str(epoch) + ".h5"
    model.save_weights(name)
    print("Saved model to disk")
    return



def on_epoch_end(epoch):
    # Function invoked at end of each epoch. Prints generated text.
    print('----- Testing entorpy after Epoch: %d' % epoch)
    entropy = 0
    batch_num = 0
    for i, batch in enumerate(get_batch(read_data(valid_path))):
        _input = batch[0]
        _labels = batch[1]
        x = model.test_on_batch(_input, _labels)
        entropy += x
        batch_num = i
    return entropy / batch_num * math.log(math.e, 2)

def readFasta(filename):
    reads = []
    with open(filename, "rU") as handle:
        for record in SeqIO.parse(handle, "fasta") :
                reads.append(record)
    return reads


model = Model_CNN_BiLSTM_noatten(5)
#model=model_CNN_LSTM()
print(model.summary())
model.load_weights("/home/cww/DeepZip/DNA/fish_xlt_final/weight_noatten_2019-10-28CNN+LSTM.h5")
optim = keras.optimizers.Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0, amsgrad=False)
model.compile(loss=loss_fn, optimizer=optim,metrics=['accuracy'])

entropy = []
# for epoch in range(epochs):
# print("this is epoch: ", epoch)

valid_inputs = []
valid_labels = []
loss_total=[]

for i, record in enumerate(readFasta(test_path)):
   train_inputs = []
   train_labels = []
   for j, batch in enumerate(get_batch(read_data(record.seq))):
        _input = batch[0]
        _labels = batch[1]
    # print(_input)
        train_inputs.append(_input)
        train_labels.append(_labels)
   train_inputs = np.array(train_inputs)
   train_labels = np.array(train_labels)
   loss, acc = model.evaluate(train_inputs, train_labels, batch_size=batch_size, verbose=0)
   print(i, '\t', record.name, '\t', len(record.seq), '\t', loss)
   loss_total.append(loss)
print("Total average is:")    
print(np.mean(loss_total))
    # x=model.train_on_batch(_input,_labels)
    # if(i%100==0):
    #     print(epoch,'\t', x*math.log(math.e,2))


# train_inputs = train_inputs.reshape(1,64,5)

# train_labels = np_utils.to_categorical(train_labels, 5)

# valid_inputs = train_inputs.reshape(len(valid_inputs),-1)
# valid_labels = np_utils.to_categorical(valid_labels, 5)


# saveModel(epoch)
# testEntropy = on_epoch_end(epoch)
# print(testEntropy)
# entropy.append(testEntropy)
# print(entropy)





