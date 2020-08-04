from __future__ import print_function
from keras.callbacks import LambdaCallback, Callback
from keras.layers import Dense, Bidirectional,Average,average,Input
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import LSTM
from keras.layers import Conv1D, MaxPooling1D,AveragePooling1D
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



train_path = "data/train.fasta"
valid_path = "data/valid.fasta"
test_path = "data/test.fasta"
todaydate=str(datetime.date.today())+'CNN+LSTM'
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

def read_data(data_path):
    text = read_fasta(data_path)
    for i in range(0, len(text) - maxlen, step):
        sentence = text[i: i + maxlen]
        next_char = text[i + maxlen]
        yield sentence, next_char
def vectorization(sentences, next_chars): 
    x = np.zeros((maxlen, len(chars)), dtype=np.bool) 
    y = np.zeros((len(chars)), dtype=np.bool)
    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            x[ t, char_indices[char]] = 1    
        y[ char_indices[next_chars[i]]] = 1 
    return x, y
def get_batch(stream):
    sentences = []
    next_chars = []
    for sentence, next_char in stream:  
        sentences.append(sentence)
        next_chars.append(next_char)
        
        data_tuple = vectorization(sentences,next_chars)
        yield data_tuple   
        sentences = []   
        next_chars = []



def Model_CNN_BiLSTM(bs,time_steps, alphabet_size):  #cnn_emerge+biLSTM+attention

   # convs = []
   # filter_sizes = [6,7,8]
    model = Sequential()
    model.add(Conv1D(filters=1024,
                     kernel_size=24,
                     trainable=True,
                     padding='valid',
                     activation='relu',
                     strides=1,input_shape=( maxlen,5)))
    model.add(MaxPooling1D(pool_size=3))
    
    model.add(BatchNormalization())

    model.add(Bidirectional(LSTM(256, stateful=False, return_sequences=True)))

    
    model.add(AttentionWithContext())
 
    model.add(Activation('relu'))

    model.add(Dense(alphabet_size, activation='softmax'))
    
    return model
def Model_CNN_BiLSTM_noBi(bs,time_steps, alphabet_size):  #cnn_emerge+biLSTM+attention

   # convs = []
   # filter_sizes = [6,7,8]
    model = Sequential()
    model.add(Conv1D(filters=1024,
                     kernel_size=24,
                     trainable=True,
                     padding='valid',
                     activation='relu',
                     strides=1,input_shape=( maxlen,5)))
    model.add(MaxPooling1D(pool_size=3))    
    model.add(BatchNormalization())    
    model.add(LSTM(256, stateful=False, return_sequences=True))
    model.add(AttentionWithContext())
    model.add(Activation('relu'))
    model.add(Dense(alphabet_size, activation='softmax'))   
    return model
    
def Model_CNN_BiLSTM_noBi_fanxiang(bs,time_steps, alphabet_size):  #cnn_emerge+biLSTM+attention

   # convs = []
   # filter_sizes = [6,7,8]
    model = Sequential()
    model.add(Conv1D(filters=1024,
                     kernel_size=24,
                     trainable=True,
                     padding='valid',
                     activation='relu',
                     strides=1,input_shape=( maxlen,5)))
    model.add(MaxPooling1D(pool_size=3))
    
    model.add(BatchNormalization())
 
    model.add(LSTM(256, stateful=False, return_sequences=True,go_backwards=True))
   
    
    
    model.add(AttentionWithContext())
  
    model.add(Activation('relu'))

    model.add(Dense(alphabet_size, activation='softmax'))
    
    return model
def Model_CNN_BiLSTM_noCNN(bs,time_steps, alphabet_size): 

   # convs = []
   # filter_sizes = [6,7,8]
    model = Sequential()  
    model.add(BatchNormalization(input_shape=( maxlen,5)))  
    model.add(Bidirectional(LSTM(256, stateful=False, return_sequences=True)))  
    model.add(AttentionWithContext())
    model.add(Activation('relu'))
   # model.add(BatchNormalization()
   # model.add(Dense(input_dim))
    #model.add(Activation('softmax'))
    model.add(Dense(alphabet_size, activation='softmax'))
    
    return model
def Model_CNN_BiLSTM_noatten(bs,time_steps, alphabet_size):  #cnn_emerge+biLSTM+attention

   # convs = []
   # filter_sizes = [6,7,8]
    model = Sequential()
    model.add(Conv1D(filters=1024,
                     kernel_size=24,
                     trainable=True,
                     padding='valid',
                     activation='relu',
                     strides=1,input_shape=( maxlen,5)))
    model.add(MaxPooling1D(pool_size=3))
    
    model.add(BatchNormalization())
  
    model.add(Bidirectional(LSTM(256, stateful=False, return_sequences=True)))
    model.add(Flatten())
    model.add(Dense(alphabet_size, activation='softmax'))
    
    return model
def model_LSTM():
    # build the model: a single LSTM
    print('Build model...')
    model = Sequential()
    model.add(LSTM(64, return_sequences=True))
    model.add(LSTM(64, return_sequences=True))
    model.add(LSTM(64))
    model.add(Dense(input_dim, activation='softmax'))
    optimizer = RMSprop(lr=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)
    return model

def saveModel(epoch):
    # serialize model to JSON
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    #serialize weights to HDF5
    name="model_"+str(epoch)+".h5"
    model.save_weights(name)
    print("Saved model to disk")
    return

model = Model_CNN_BiLSTM(batch_size,maxlen, input_dim)
print(model.summary())

def fit_model(X, Y,X_valid,Y_valid, bs, nb_epoch, model):
    y = Y
    optim = keras.optimizers.Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0, amsgrad=False)
    model.compile(loss=loss_fn, optimizer=optim,metrics=['accuracy'])
    checkpoint = ModelCheckpoint('cpt/weight_CNN+LSTM_'+todaydate+'.h5', monitor='loss', verbose=1, save_best_only=True, mode='min',
                                 save_weights_only=True)
    csv_logger = CSVLogger('log/log_'+todaydate, append=True, separator=';')
    early_stopping = EarlyStopping(monitor='loss', mode='min', min_delta=0.005, patience=3, verbose=1)
    tensorboad = TensorBoard(log_dir='board/board_'+todaydate)
    callbacks_list = [checkpoint, csv_logger, early_stopping,tensorboad]
    # callbacks_list = [checkpoint, csv_logger]
    model.fit(X, y, epochs=nb_epoch, batch_size=bs, verbose=1,validation_data=(X_valid,Y_valid), shuffle=True, callbacks=callbacks_list)
entropy = []

train_inputs=[]
print(str(np.array(train_inputs).shape))
train_labels=[]
valid_inputs=[]
valid_labels=[]
for i, batch in enumerate(get_batch(read_data(train_path))):
    _input = batch[0]  
    _labels = batch[1]
    #print(_input)
    train_inputs.append(_input)
    train_labels.append(_labels)
 
print("train end")
for i, batch in enumerate(get_batch(read_data(valid_path))):
    _input = batch[0]  
    _labels = batch[1]
    valid_inputs.append(_input)
    valid_labels.append(_labels)
    
print("valid end")
train_inputs=np.array(train_inputs)
print(str(train_inputs.shape))
train_labels=np.array(train_labels)
valid_inputs=np.array(valid_inputs)
valid_labels=np.array(valid_labels)
fit_model(train_inputs, train_labels,valid_inputs,valid_labels, batch_size,epochs , model)





