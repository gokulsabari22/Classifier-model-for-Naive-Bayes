# Import Libraries
import pandas as pd
import numpy as np
import os
from sklearn.utils import shuffle
from nltk.corpus import stopwords
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV
import pickle

# Import Data
random_state = 0
df = pd.read_csv("dataset.csv")


# Drop extra columns
df = df.drop("Unnamed: 0", axis = 1)

# Check for null values
null = df.isnull().sum()


# Preprocessing the data using NLTK Toolkit
corpus = []
for i in range(len(df)):
    stop_words = set(stopwords.words("english"))
    sentence = re.sub(r"(\\n|\\t)", " ",df['Text'][i])
    sentence = re.sub("[^A-Za-z]"," ", sentence)
    sentence = sentence.lower()
    sentence = sentence.split()
    sentence = [words for words in sentence if words not in stop_words]
    sentence = ' '.join(sentence)
    corpus.append(sentence)


# Seperate X and Y values
vector = CountVectorizer()
X = vector.fit_transform(corpus)
y = df["Label"]


# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = random_state)

# Fit the model
clf = MultinomialNB(alpha = 1.0)
clf.fit(X_train, y_train)

# Predict and evaluate the model
predict = clf.predict(X_test)
print(clf.score(X_train, y_train)*100)
print(clf.score(X_test, y_test)*100)
print(classification_report(y_test, predict))
tn, fp, fn, tp = (confusion_matrix(y_test, predict)).ravel()
print('TN, FP, FN, TP :', tn, fp, fn, tp)

# Grid Search CV for hyper-parameter tuning
parameters = [{'alpha':[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]}]
grid_search = GridSearchCV(estimator = clf,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
grid_search = grid_search.fit(X_train, y_train)
print(grid_search.best_params_)

# Fit the tuned aplha
clf = MultinomialNB(alpha = 0.1)
clf.fit(X_train, y_train)

# Evaluate the tuned model
predict = clf.predict(X_test)
print(clf.score(X_train, y_train)*100)
print(clf.score(X_test, y_test)*100)
print(classification_report(y_test, predict))
tn, fp, fn, tp = (confusion_matrix(y_test, predict)).ravel()
print('TN, FP, FN, TP :', tn, fp, fn, tp)

# Pickle the model
with open("Model", "wb") as f:
    pickle.dump(clf, f)

with open("Vector", "wb") as fh:
    pickle.dump(vector, fh)

