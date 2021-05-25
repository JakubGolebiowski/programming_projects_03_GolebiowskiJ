# -------------------------
# Jakub Gołębiowski
# start:
# 30.04.2021
# last changes:
# 25.05.2021
# -------------------------

# Load datasets like iris (3-class dataset)
from sklearn import datasets
# Manipulate data
from sklearn.model_selection import train_test_split
# SVM
from sklearn import svm
# DECISION TREE
from sklearn.tree import DecisionTreeClassifier
# random forest
from sklearn.ensemble import RandomForestClassifier

# MLPClassifier
from sklearn.neural_network import MLPClassifier
# Gradient Boosting Classifier
from sklearn.ensemble import GradientBoostingClassifier
# numpy (arrays and matrix magic etc.)
import numpy as np
import pandas as pd
import datetime
import copy
import matplotlib.pyplot as plt

# using for multi thread processing - 20-30% faster
from joblib import parallel_backend

# from numba import jit, njit, vectorize

# Generating satisfied data format
# # print("Divide the data and separate the inputs")
# data = pd.read_csv("4BitClasses.txt", sep=";", dtype={"inputs": np.str})
# data2 = data['inputs'].apply(lambda x: pd.Series(list(x)))
# data = data['class']
# # data2 = data2.join(data["class"])
# print(data.head())
# print(data2.head())
# # data = pd.merge(data["class"], data2)
# # print(data.head())
# data.to_csv('data_Y2.csv', index=False)
# data2.to_csv('data_X2.csv', index=False)
# # data3 = pd.concat([data, data2], axis=1)
# # print(data3.head())
# # data2.to_csv('data_merged.csv', index=False)

# importing data from file
X = pd.read_csv("data_X2.csv")
y = pd.read_csv("data_Y2.csv")
print(X.shape)
print(y.shape)


# spliting data into test and train
def load_data(test_size=0.90, random_state=0):
    X = pd.read_csv("data_X2.csv")
    y = pd.read_csv("data_Y2.csv")

    # higher preprocessing data/sorting at least one member of each one for learn
    # xy = pd.concat([X, y], axis=1)
    # print(y.head())
    # print(X.head())
    # print(xy.head())
    # itera = xy['class'].unique()
    # new_data = []
    # for cls in range(0, len(itera)):
    #     for index, row in xy.iterrows():
    #         if itera[cls] == row['class']:
    #             new_data.append(index)
    #             # print(cls, index)
    #             cls += 1
    #         if cls == 222:
    #             break
    #     if cls == 222:
    #         break
    #         break
    # new_x = X
    # new_y = y
    # for i in range(1, len(new_data)):
    #     temp_x = new_x.loc[i].copy()
    #     temp_y = new_y.loc[i].copy()
    #     print(temp_x,new_x.loc[i], temp_y, new_x.loc[i])
    #     new_x.loc[i]= new_x.loc[new_data[i]].copy()
    #     new_x.loc[new_data[i]] = temp_x
    #     new_y.loc[i] = new_y.loc[new_data[i]].copy()
    #     new_y.loc[new_data[i]] = temp_y
    #     print(temp_x, new_x.loc[i], temp_y, new_x.loc[i])

    # new_x.to_csv('data_X2.csv', index=False)
    # new_y.to_csv('data_Y2.csv', index=False)
    # print(new_data.shape)
    # print(new_data)
    # dividing X, y into train and test data

    # splitting output
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


X_train, X_test, y_train, y_test = load_data()
y_train = y_train.values.ravel()


# SVM - poorly working for classification so we didn't use it
def svm_our():
    clf = svm.SVC(gamma='auto')
    clf.fit(X_train, y_train)

    print(clf.predict(X_test))
    print(y_test)
    print(clf.score(X_test, y_test))


# Decision Tree - poorly working for classification so we didn't use it
def decision_tree_our(X_train, X_test, y_train, y_test, max_depth=2):
    d_tree_model = DecisionTreeClassifier(max_depth=max_depth, criterion='entropy', ).fit(X_train, y_train)
    d_tree_predictions = d_tree_model.predict(X_test)
    #
    # print(d_tree_predictions)
    return d_tree_model.score(X_test, y_test)


# Random Forest - poorly working for classification so we didn't use it
def random_tree(X_train, X_test, y_train, y_test, max_depth=2):
    d_tree_model = RandomForestClassifier(max_depth=max_depth, criterion='entropy', ).fit(X_train, y_train)
    d_tree_predictions = d_tree_model.predict(X_test)
    #
    # print(d_tree_predictions)
    return d_tree_model.score(X_test, y_test)


# MLP Classifier - our main subject of research
def mlp_classifier_our(activation, X_train, X_test, y_train, y_test, hidden_layer_sizes, max_iter=300):
    with parallel_backend('threading', n_jobs=-1):
        clf = MLPClassifier(random_state=404, hidden_layer_sizes=hidden_layer_sizes, max_iter=max_iter,
                            activation=activation).fit(X_train, y_train)
    # clf.predict_proba(X_test[:1])

    # print(clf.predict(X_test))
    # print(y_test)

    return clf.score(X_test, y_test)


# Gradient Boosting - poorly working for classification so we didn't use it
def gradient_boosting_our(X_train, X_test, y_train, y_test, n_estimators=100):
    with parallel_backend('threading', n_jobs=-1):
        clf = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=1.0,
                                         max_depth=1, random_state=0).fit(X_train, y_train)
    # print(clf.predict(X_test))
    # print(y_test)
    abx = clf.score(X_test, y_test)
    print(abx)
    return abx


# svm_our()
# decision_tree_our()
# mlp_classifier_our()
# gradient_boosting_our()

# MLP starts here - once it runs it generates scores of 33 models * len(iters_tab) in plots and csv
def MLP():
    iters_tab = [150, 400, 500]
    for max_iter2 in iters_tab:
        tab = ["relu", "logistic", "tanh"]
        train_to_test = [5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90]
        datas = pd.DataFrame()
        maintab = []
        max_iter = max_iter2
        for activation in tab:
            tx = []
            ty = []

            for i in train_to_test:
                print([activation, i])
                X_train, X_test, y_train, y_test = load_data(test_size=i / 100)
                y_train = y_train.values.ravel()
                score = mlp_classifier_our(activation, X_train, X_test, y_train, y_test, hidden_layer_sizes=(64, 64),
                                           max_iter=max_iter)
                tx.append(i / 100)
                ty.append(score)

            maintab.append(ty)
            plt.plot(tx, ty)
            plt.xlabel("Train to test")
            plt.ylabel("Scoring")
            plt.title(str(activation) + " Iterations: " + str(max_iter))
            plt.savefig(str(activation) + "_sorted_64_64_" + "_" + str(max_iter) + ".png")
            plt.close()

        maintab = np.transpose(maintab)
        # print(maintab)

        datas = pd.DataFrame(maintab, columns=["relu" + str(max_iter), "logistic" + str(max_iter), "tanh" + str(max_iter)])
        #
        # print(datas)

        datas.to_csv(str(max_iter) + "_sorted_" + str(datetime.datetime.now().timestamp()) + "_.csv", index=False)


# start for research of Decision trees
def DecisionTree():
    tab = [5000, 1000, None]
    train_to_test = [5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90]
    datas = pd.DataFrame()
    maintab = []
    # max=400
    for max_depth in tab:
        tx = []
        ty = []

        for i in train_to_test:
            print([max_depth, i])
            X_train, X_test, y_train, y_test = load_data(test_size=i / 100)
            y_train = y_train.values.ravel()
            score = decision_tree_our(X_train, X_test, y_train, y_test, max_depth)
            tx.append(i / 100)
            ty.append(score)

        maintab.append(ty)
        plt.plot(tx, ty)
        plt.xlabel("Train to test")
        plt.ylabel("Scoring")
        plt.title("Max depth: " + str(max_depth))
        plt.savefig("DecisionTree_" + str(max_depth) + ".png")
        plt.close()

    maintab = np.transpose(maintab)
    # print(maintab)

    datas = pd.DataFrame(maintab, columns=[str(tab[0]), str(tab[1]), str(tab[2])])
    #
    # print(datas)

    datas.to_csv("decisionTree_" + str(datetime.datetime.now().timestamp()) + "_.csv", index=False)


# start for research of Random Forest
def RandomTreeClass():
    tab = [5000, 1000, None]
    train_to_test = [5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90]
    datas = pd.DataFrame()
    maintab = []
    # max=400

    for max_depth in tab:
        tx = []
        ty = []

        for i in train_to_test:
            print([max_depth, i])
            X_train, X_test, y_train, y_test = load_data(test_size=i / 100)
            y_train = y_train.values.ravel()
            score = random_tree(X_train, X_test, y_train, y_test, max_depth)
            tx.append(i / 100)
            ty.append(score)

        maintab.append(ty)
        plt.plot(tx, ty)
        plt.xlabel("Train to test")
        plt.ylabel("Scoring")
        plt.title("Max depth: " + str(max_depth))
        plt.savefig("RandomTree_" + str(max_depth) + ".png")
        plt.close()

    maintab = np.transpose(maintab)
    # print(maintab)

    datas = pd.DataFrame(maintab, columns=[str(tab[0]), str(tab[1]), str(tab[2])])
    #
    # print(datas)

    datas.to_csv("RandomTree_" + str(datetime.datetime.now().timestamp()) + "_.csv", index=False)


# Start for Gradient Boost - we didn't have time to examine the examples
def GradientBoost():
    tab = [20, 50, 100]
    train_to_test = [5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90]
    datas = pd.DataFrame()
    maintab = []
    # max=400
    for n_estimators in tab:
        tx = []
        ty = []

        for i in train_to_test:
            print([n_estimators, i])
            X_train, X_test, y_train, y_test = load_data(test_size=i / 100)
            y_train = y_train.values.ravel()
            score = gradient_boosting_our(X_train, X_test, y_train, y_test, n_estimators)
            tx.append(i / 100)
            ty.append(score)

        maintab.append(ty)
        plt.plot(tx, ty)
        plt.xlabel("Train to test")
        plt.ylabel("Scoring")
        plt.title("Max depth: " + str(n_estimators))
        plt.savefig("GradientBoost_" + str(n_estimators) + ".png")
        plt.close()

    maintab = np.transpose(maintab)
    # print(maintab)

    datas = pd.DataFrame(maintab, columns=[str(tab[0]), str(tab[1]), str(tab[2])])
    #
    # print(datas)

    datas.to_csv("gradientBoost_" + str(datetime.datetime.now().timestamp()) + "_.csv", index=False)


MLP()
# DecisionTree()
# GradientBoost()
# RandomTreeClass()
