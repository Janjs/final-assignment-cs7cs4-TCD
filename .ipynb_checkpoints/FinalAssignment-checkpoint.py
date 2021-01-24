import matplotlib
import numpy as np
import pandas as pd
import json_lines
import nltk
import re
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import WhitespaceTokenizer
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split, cross_val_score

def readData():
    X = []; y = []; z = []
    with open('reviews_175.jl', 'rb') as f:
        for item in json_lines.reader(f):
            X.append(item['text'])
            y.append(item['voted_up'])
            z.append(item['early_access'])
    return [X, y, z];


def extractTextFeatures(X):
    tokenizer = CountVectorizer().build_tokenizer()
    #print(WhitespaceTokenizer().tokenize("Here’s example text, isn’t it?"))
    #print(word_tokenize("Here's example text, isn't it"))

    #print(tokenizer("likes liking liked"))
    #print(WhitespaceTokenizer().tokenize("likes liking liked"))
    #print(word_tokenize("likes liking liked"))

    stemmer = PorterStemmer()
    tokens = word_tokenize("likes liking liked")
    stems = [stemmer.stem(token) for token in tokens]
    print(stems)

    docs = ['This is the first document., '
            'This is the second second document.',
            'And the third one.',
            'Is this the first document?']

    # nltk.download('stopwords')
    # vectorizer = CountVectorizer(stop_words=nltk.corpus.stopwords.words('english'), min_df=5)
    # X = vectorizer.fit_transform(X)
    # print(vectorizer.get_feature_names())
    # print(X.toarray())

    vectorizer = TfidfVectorizer(min_df=2)
    X = vectorizer.fit_transform(X)
    print(vectorizer.get_feature_names())

    print(X.shape)

    f = open("newDataset.txt", "a")
    f.write(str(X.toarray()))
    f.close()

    return X


def naiveBayesModel(X_train, X_test, y_train, y_test):
    nbModel = MultinomialNB().fit(X_train, y_train)
    predictions = nbModel.predict(X_test)
    accuracy = np.mean(predictions == y_test)
    print("Accuracy NB: "+str(accuracy*100)+"%")

def svmModel(X_train, X_test, y_train, y_test):
    svmModel = SGDClassifier().fit(X_train, y_train)
    predictions = svmModel.predict(X_test)
    accuracy = np.mean(predictions == y_test)
    print("Accuracy SVM: "+str(accuracy*100)+"%")


if __name__ == "__main__":
    X, y, z = readData()

    data = np.column_stack((X, y, z))
    print(len(X))

    X = extractTextFeatures(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    naiveBayesModel(X_train, X_test, y_train, y_test)
    svmModel(X_train, X_test, y_train, y_test)

