'''Author: Rongjie Wang'''

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

# for reproducibility
#train_path = '/home/cww/DeepZip/rongjiewang/DNN_dataset/train.fasta'
#valid_path = '/home/cww/DeepZip/rongjiewang/DNN_dataset/valid.fasta'
#test_path = '/home/cww/DeepZip/rongjiewang/DNN_dataset/test.fasta'
train_path = "/home/cww/DeepZip/DNA/fish_xlt_final/train.fasta"
valid_path = "/home/cww/DeepZip/DNA/fish_xlt_final/valid.fasta"
test_path = "/home/cww/DeepZip/DNA/fish_xlt_final/test.fasta"
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


def loadModel():
    #model.load_weights('my_model_weights.h5')
    #json and create model
    json_file = open('./model/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights("./model/model.h5")
    print("Loaded model from disk")
    return model


def model_patern():
    # build the model: a single LSTM
    print('Build model...')
    model = Sequential()
    model.add(Conv1D(filters=1024,
                     kernel_size=24,
                     #kernel_initializer=my_kernel_initializer,
                     #trainable=False,
                     #padding='same',
                     #activation=None,
                     #use_bias=False,
                     #bias_initializer= keras.initializers.Constant(value=-7),
                     strides=1))
    model.add(MaxPooling1D(pool_size=3))
    model.add(Dropout(0.1))
    model.add(LSTM(256,return_sequences=True))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dense(input_dim))
    model.add(Activation('softmax'))
    optimizer = RMSprop(lr=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)
    return model

def model_CNN_LSTM():
    # build the model: a single LSTM
    print('Build model...')
    model = Sequential()
    model.add(Conv1D(filters=1024,
                     kernel_size=24,
                     trainable=True,
                     padding='valid',
                     activation='relu',
                     strides=1,input_shape=( maxlen,5)))
    model.add(MaxPooling1D(pool_size=3))
    model.add(Dropout(0.1))
    model.add(LSTM(256,return_sequences=True))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dense(input_dim))
    model.add(Activation('softmax'))
    # optimizer = RMSprop(lr=0.001)
    # model.compile(loss='categorical_crossentropy', optimizer=optimizer)
    return model
    



def model_CNN():
    # build the model: a single LSTM
    print('Build model...')
    model = Sequential()
    model.add(Conv1D(filters=320,
                     kernel_size=6,
                     trainable=True,
                     padding='valid',
                     activation='relu',
                     strides=1))
    model.add(MaxPooling1D(pool_size=3,strides=3))
    model.add(Dropout(0.1))

    model.add(Conv1D(filters=480,
                     kernel_size=4,
                     trainable=True,
                     padding='valid',
                     activation='relu',
                     strides=1))
    model.add(MaxPooling1D(pool_size=2,strides=2))
    model.add(Dropout(0.1))
    model.add(Conv1D(filters=960,
                     kernel_size=4,
                     trainable=True,
                     padding='valid',
                     activation='relu',
                     strides=1))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dense(input_dim))
    model.add(Activation('softmax'))
    optimizer = RMSprop(lr=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)
    return model
def Model_emerge(bs,time_steps, alphabet_size):  #cnn_emerge+biLSTM+attention

    convs = []
    filter_sizes = [6,7,8]
    #input_sequence=Input(shape=(64,5))
    #model = Sequential()
    #embedded_sequences = Embedding(alphabet_size, 16, batch_input_shape=(bs, time_steps))(input_sequence)
   # print("embedded shape",embedded_sequences.shape)
    #for fsz in filter_sizes:
    #    l_conv = Conv1D(nb_filter=512,filter_length=fsz,activation='relu')(input_sequence)
    #    print("l_conv"+str(fsz),l_conv.shape)
    #    l_pool = AveragePooling1D(pool_size=3)(l_conv)
    #    print("l_pool"+str(fsz),l_pool.shape)
    #    l_dense=Dense(64, activation='relu')(l_pool)
    #    convs.append(l_dense)
   # m_merge=Average()(convs)
    
  #  model = Sequential()
  #  model.add(Embedding(alphabet_size, 16, batch_input_shape=(bs, time_steps)))
   ## model.add(average([model_1,model_2,model_4,model_8,model_16]))  
    #biLSTM=Bidirectional(LSTM(128, stateful=False, return_sequences=True))(m_merge)
  #  model.add(Bidirectional(CuDNNLSTM(32, stateful=False, return_sequences=False)))
 #   model.add(Dropout(0.2))
 #   atten=AttentionWithContext()(biLSTM)
 #   att=Dropout(0.2)(atten)
   # att=Flatten()(att)
  #  att=Dense(512)(att)
 #   att=Activation('relu')(att)
   

  #  att=BatchNormalization()(att)
  #  output=Dense(alphabet_size, activation='softmax')(att)
   # model= Model(inputs=input_sequence, outputs=output)
  #  return model
    
    
    model = Sequential()
    model.add(Embedding(alphabet_size, 16, batch_input_shape=(bs, time_steps,5)))
    model.add(Bidirectional(LSTM(128, stateful=False, return_sequences=True)))
    # model.add(Bidirectional(CuDNNLSTM(32, stateful=False, return_sequences=False)))
    model.add(Dropout(0.2))
    #model.add(BatchNormalization())
    model.add(AttentionWithContext())
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(512))
    
    #model.add(PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=None))

    model.add(BatchNormalization())
    model.add(Dense(alphabet_size, activation='softmax'))
    return model
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
    #model.add(Dropout(0.5))
    #model.add(Embedding(alphabet_size, 16, input_length= 5))
    model.add(LSTM(256, stateful=False, return_sequences=True,go_backwards=True))
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
def Model_CNN_BiLSTM_noCNN(bs,time_steps, alphabet_size): 

   # convs = []
   # filter_sizes = [6,7,8]
    model = Sequential()

    
    model.add(BatchNormalization(input_shape=( maxlen,5)))
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

model = Model_CNN_BiLSTM(128,64, input_dim)
print(model.summary())
def on_epoch_end(epoch):
    # Function invoked at end of each epoch. Prints generated text.
    print('----- Testing entorpy after Epoch: %d' % epoch)
    entropy = 0
    batch_num = 0
    for i, batch in enumerate(get_batch(read_data(valid_path))):  
        _input = batch[0]
        _labels = batch[1]
        x=model.test_on_batch(_input,_labels)
        entropy += x
        batch_num = i
    return entropy/batch_num*math.log(math.e, 2)   
def fit_model(X, Y,X_valid,Y_valid, bs, nb_epoch, model):
    y = Y
    optim = keras.optimizers.Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0, amsgrad=False)
    model.compile(loss=loss_fn, optimizer=optim,metrics=['accuracy'])
    checkpoint = ModelCheckpoint('/home/cww/DeepZip/DNA/fish_xlt_final/weight_noAtten_justlast_'+todaydate+'.h5', monitor='loss', verbose=1, save_best_only=True, mode='min',
                                 save_weights_only=True)
    csv_logger = CSVLogger('/home/cww/DeepZip/DNA/fish_xlt_final/log_file_noAtten_justlast_'+todaydate, append=True, separator=';')
    early_stopping = EarlyStopping(monitor='loss', mode='min', min_delta=0.005, patience=3, verbose=1)
    tensorboad = TensorBoard(log_dir='/home/cww/DeepZip/DNA/fish_xlt_final/tensor_noAtten_justlast_'+todaydate)
    callbacks_list = [checkpoint, csv_logger, early_stopping,tensorboad]
    # callbacks_list = [checkpoint, csv_logger]
    model.fit(X, y, epochs=nb_epoch, batch_size=bs, verbose=1,validation_data=(X_valid,Y_valid), shuffle=True, callbacks=callbacks_list)
entropy = []
#for epoch in range(epochs):
# print("this is epoch: ", epoch)
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
    
    #x=model.train_on_batch(_input,_labels)  
    # if(i%100==0):
    #     print(epoch,'\t', x*math.log(math.e,2))
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





