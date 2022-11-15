import sys
import pandas as pd
import numpy as np
import seaborn as sns
import statistics as sts
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn import metrics
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.datasets import load_svmlight_file
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

def find_best_k_value(X, y):
    X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size=0.5, random_state = 42)

    scaler = preprocessing.MaxAbsScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)

    # Get the best K value
    acc = []
    for i in range(1, 40, 2):
        neigh = KNeighborsClassifier(n_neighbors = i).fit(X_train,y_train)
        y_pred = neigh.predict(X_test)
        acc.append(metrics.accuracy_score(y_test, y_pred))

    # Plot accuracy for each k-neighbor classification    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 40, 2), acc, color = 'blue', linestyle='dashed', 
        marker='o', markerfacecolor='red', markersize=10)
    plt.title('Accuracy vs. K Value')
    plt.xlabel('K')
    plt.ylabel('Accuracy')
    plt.grid()
    plt.savefig('./images/k-neighbors.svg')
    plt.clf()

    best_k = acc.index(max(acc)) + 1
    print('Best K:', best_k)
    print('Accuracy for this K =', acc.index(max(acc)) + 1, 
                ': ', max(acc))

    return best_k


def main(fname):
    # Create KNN classifiers for each random_state
    print('Loading data...')
    X, y = load_svmlight_file(fname)

    print('Finding best K value...')
    k = find_best_k_value(X, y)

    random_states = [5, 8, 13, 42, 99]

    acc = []
    dfs = []
    scores = []

    print('Testing classifiers for different data splits...')
    for random_state in random_states:
        X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size=0.5, random_state = random_state)

        scaler = preprocessing.MaxAbsScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.fit_transform(X_test)

        knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
        knn.fit(X_train, y_train)

        y_pred = knn.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)

        acc.append(knn.score(X_test, y_test))
        dfs.append(pd.DataFrame(cm))
        scores.append(f1_score(y_test, y_pred, average=None))


    df_mean = pd.DataFrame(dfs[0] + dfs[1] + dfs[2] + dfs[3] + dfs[4])/5

    plt.figure()
    ax = sns.heatmap(df_mean.to_numpy(), annot=True, fmt='.4', cmap='Blues')
    ax.set(xlabel='Predicted', ylabel='True Label')
    plt.savefig('./images/confusion_matrix.svg')

    scores_df = pd.DataFrame(scores).transpose()
    scores_df['mean'] = scores_df.mean(axis=1)

    with open('./data/f1_score.txt', 'w') as f:
        f.write('Accuracy: %.2f\n' % sts.mean(acc))
        f.write('f1 score:\n')
        f.write(str(scores_df['mean']))


if __name__ == "__main__":
    if len(sys.argv) != 2:
            sys.exit("Use: generate_report.py <file>")

    main(sys.argv[1])
