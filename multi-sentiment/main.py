import sklearn

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
import autosklearn
import re
import auto
import lstm

if __name__ == '__main__':
    # Load data
    trainData = pd.read_csv("Headline_Trainingdata.csv")
    testData = pd.read_csv("Headline_Testingdata.csv")
    # Features and labels
    X, Y = trainData['text'], trainData['sentiment']

    # Preprocess the data
    # Transform texrs features to nums
    X = X.apply(lambda x: x.lower())
    X = X.apply((lambda x: re.sub('[^a-zA-z0-9\s]', '', x)))

    max_features = 1000
    tokenizer = Tokenizer(num_words=max_features, split=' ')
    # X_train and X_test
    tokenizer.fit_on_texts(X.values)
    X = tokenizer.texts_to_sequences(X.values)
    X = pad_sequences(X)

    Y = pd.get_dummies(Y).values


    m = lstm.lstm(max_features, X)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

    m.train(X_train, Y_train, X_test, Y_test)


    # automl = autosklearn.classification.AutoSklearnClassifier()
    # automl.fit(X_train, Y_train)
    # Y_hat = automl.predict(X_test)
    # print("Accuracy score", sklearn.metrics.accuracy_score(Y_test, Y_hat))