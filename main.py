import pandas as pd
from keras.preprocessing import text, sequence
from keras import layers, models, optimizers
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB



# Load Datasets to Data Frames
trainFilepath = 'data/train.csv'
testFilepath = 'data/test.csv'
trainData = pd.read_csv(trainFilepath, names=['id', 'title', 'abstract', 'introduction', 'label'])
testData = pd.read_csv(testFilepath, names=['id', 'title', 'abstract', 'introduction', 'label'])


# Split into Train and Validate Datasets
train, valid = train_test_split(trainData)

# Label Encoder
encoder = LabelEncoder()
train_encodedLabels = encoder.fit_transform(train['label'])
valid_encodedLabels = encoder.fit_transform(valid['label'])



# Count Vectors:
# Train: Create and Transform Count vectorizer object 
trainCount_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
trainCount_vect.fit(train['abstract'])
# print(trainCount_vect.vocabulary_)
# print("\n\n")
trainCount_vect.fit(train['title'])
# print(trainCount_vect.vocabulary_)
trainCount_vect.fit_transform(train['introduction'])
valid_count = trainCount_vect.transform(valid)


# Test: Create and Transform Count vectorizer object 
testCount_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
testCount_vect.fit(train['abstract'])
# print(testCount_vect.vocabulary_)
# print("\n\n")
testCount_vect.fit(train['title'])
# print(testCount_vect.vocabulary_)
testCount_vect.fit_transform(train['introduction'])



# TF-IDF:
# Word Level TF-IDFs
title_tfidf = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}')
abstract_tfidf = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}')
intro_tfidf = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}')
title_tfidf.fit_transform(train['title'])
title_valid_tfidf = title_tfidf.transform(valid['title'])
abstract_tfidf.fit_transform(train['abstract'])
abstract_valid_tfidf = abstract_tfidf.transform(valid['abstract'])
intro_tfidf.fit_transform(train['introduction'])
intro_valid_tfidf = intro_tfidf.transform(valid['introduction'])


# TODO: add ngram level and character levels?


# Function: Train Model
def train_model(classifier, feature_vector_train, label, feature_vector_valid, is_neural_net=False):
    # fit the training dataset on the classifier
    classifier.fit(feature_vector_train, label)
    
    # predict the labels on validation dataset
    predictions = classifier.predict(feature_vector_valid)
    
    if is_neural_net:
        predictions = predictions.argmax(axis=-1)
    
    return accuracy_score(predictions, valid_encodedLabels)


# Naive Bayes Classification:
# NB Count Vector
accuracy = train_model(MultinomialNB(), trainCount_vect, train_encodedLabels, valid_count)
print("NB, Count Vectors: ", accuracy)


