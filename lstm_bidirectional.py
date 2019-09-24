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
import pickle
#from keras.datasets import imdb
#from keras.optimizers import Adam
from word_model2embeding_matrix import make_deepLearn_data

def split_word(txt):
    if isinstance(txt, str):
        result = txt[:-1].split(' ')
    else:
        result = []
    return result

with open('word2idx_embedMatrix.pkl', 'rb') as f:
    word2idx, embedMatrix = pickle.load(f)
data_train = pd.read_pickle('Train_Data_sen_vec_Tencent.pkl')
print('Loading data...')
x_txt = data_train.txt_split.apply(split_word)

y = data_train.negative.values
X, max_len = make_deepLearn_data(x_txt, word2idx)

X_train = X[:-250]
y_train = y[:-250]

y_test = y[-250:]
X_test = X[-250:]
nb_words = len(word2idx.keys()) + 1
learning_rate = 0.01

model = Sequential()
model.add(Embedding(nb_words, embedMatrix.shape[1], weights=[embedMatrix],
                    input_length=max_len,mask_zero=True, trainable=False))
model.add(Bidirectional(LSTM(64,return_sequences=True,)))
model.add(Bidirectional(LSTM(64)))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# try using different optimizers and different optimizer configs
model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])

print('Train...')
model.fit(X_train, y_train,
          batch_size=32,
          epochs=10,
          validation_data=[X_test, y_test])

#model.save('my_model.h5')

#cost = model.train_on_batch(X_batch, Y_batch)

model.evaluate(X_test, y_test, batch_size=y_test.shape[0], verbose=False)
