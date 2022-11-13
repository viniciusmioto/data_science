#!/usr/bin/python
# -*- encoding: iso-8859-1 -*-

import sys
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_svmlight_file


def main(fname):

        if fname[-3:-1] == 'csv':
                df = pd.read_csv(fname)
                X = np.array(df.drop(['label'], 1))
                y = np.array(df['label'])
        else:
                X, y = load_svmlight_file(fname)

        print ("Spliting data...")
        X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size=0.5, random_state = 5)

        scaler = preprocessing.MaxAbsScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.fit_transform(X_test)

        # cria um kNN
        neigh = KNeighborsClassifier(n_neighbors=3, metric='euclidean')

        print ('Fitting knn')
        neigh.fit(X_train, y_train)

        # predicao do classificador
        print ('Predicting...')
        y_pred = neigh.predict(X_test)

        # mostra o resultado do classificador na base de teste
        print ('Accuracy: ',  neigh.score(X_test, y_test))

        # cria a matriz de confusao
        cm = confusion_matrix(y_test, y_pred)
        print (cm)
        print(classification_report(y_test, y_pred))


if __name__ == "__main__":
        if len(sys.argv) != 2:
                sys.exit("Use: knn.py <file>")

        main(sys.argv[1])

