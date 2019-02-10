import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
import re
from pathlib import Path
import csv

class lstm:
    def __init__(self, max_features, X):
        embed_dim = 128
        lstm_out = 196

        model = Sequential()
        model.add(Embedding(max_features, embed_dim, input_length=X.shape[1]))
        model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
        model.add(Dense(5, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        self.model = model


    def train(self, X_train, Y_train, X_test, Y_test):
        my_file = Path("model.h5")
        if my_file.is_file():
            self.model.load_weights('model.h5')
            return
        batch_size = 32
        es = EarlyStopping(monitor='acc', min_delta=0.001, patience=3)
        self.model.fit(X_train, Y_train, epochs=128, batch_size=batch_size, verbose=2, callbacks=[es])

        score, acc = self.model.evaluate(X_test, Y_test, verbose=2, batch_size=batch_size)
        print("score: %.2f" % (score))
        print("acc: %.2f" % (acc))

    def save(self):

        # serialize model to JSON
        model_json = self.model.to_json()
        with open("model.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.model.save_weights("model.h5")
        print("Saved model to disk")

    

    def predict(self,X):
        Y = self.model.predict(X)
        with open('out.csv','w') as csvfile:
            fieldnames = ['id', 'sentiment']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for idx, val in enumerate(Y):
                writer.writerow({'id': idx, 'sentiment': np.argmax(val)})
 
        
            