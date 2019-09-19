'''
#Trains a Bidirectional LSTM on the IMDB sentiment classification task.

Output after 4 epochs on CPU: ~0.8146
Time per epoch on CPU (Core i7): ~150s.
'''

from __future__ import print_function
#import numpy as np
import pandas as pd

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional
#from keras.datasets import imdb
#from keras.optimizers import Adam

max_features = 40000
# cut texts after this number of words
# (among top max_features most common words)
maxlen = 200
batch_size = 32

data_train = pd.read_pickle('Train_Data_sen_vec_Tencent.pkl')

print('Loading data...')
x_train = data_train.txt_sentence_vec.values[:-250]
y_train = data_train.negative.values[:-250]

x_test = data_train.txt_sentence_vec.values[-250:]
y_test = data_train.negative.values[-250:]

print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)


model = Sequential()
model.add(Embedding(max_features, 300, input_length=maxlen,mask_zero=True))
model.add(Bidirectional(LSTM(64,return_sequences=True,)))
model.add(Bidirectional(LSTM(64)))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# try using different optimizers and different optimizer configs
model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])

print('Train...')
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=100,
          validation_data=[x_test, y_test])

#model.save('my_model.h5')

#cost = model.train_on_batch(X_batch, Y_batch)

model.evaluate(x_test, y_test, batch_size=y_test.shape[0], verbose=False)
