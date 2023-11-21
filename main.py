import pandas as pd
from keras.preprocessing import text, sequence
from keras import layers, models, optimizers
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import sklearn.naive_bayes
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression



# Load Datasets to Data Frames
trainFilepath = 'data/train.csv'
testFilepath = 'data/test.csv'
trainData = pd.read_csv(trainFilepath, names=['id', 'title', 'abstract', 'introduction', 'label'])
testData = pd.read_csv(testFilepath, names=['id', 'title', 'abstract', 'introduction', 'label'])
testIntro = testData['introduction']

# Split into Train and Validate Datasets
trainTitle, validTitle, trainAbs, validAbs, trainIntro, validIntro, trainLabel, validLabel = train_test_split(trainData['title'], trainData['abstract'], trainData['introduction'], trainData['label'])

# Label Encoder
encoder = LabelEncoder()
train_encodedLabels = encoder.fit_transform(trainLabel)
valid_encodedLabels = encoder.fit_transform(validLabel)



# Count Vectors:
# Train: Create and Transform Count vectorizer object 
trainCount_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
trainCount_vect.fit(trainTitle)
trainCount_vect.fit(trainAbs)
# print(trainCount_vect.vocabulary_)
# print("\n\n")

# print(trainCount_vect.vocabulary_)
trainCount_vect.fit(trainIntro)
trainTitle_count = trainCount_vect.transform(trainTitle)
validTitle_count = trainCount_vect.transform(validTitle)
train_abs_count = trainCount_vect.transform(trainAbs)
valid_abs_count = trainCount_vect.transform(validAbs)



# # Test: Create and Transform Count vectorizer object 
# testCount_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
# testCount_vect.fit(trainAbs)
# # print(testCount_vect.vocabulary_)
# # print("\n\n")
# testCount_vect.fit(trainTitle)
# # print(testCount_vect.vocabulary_)
# testCount_vect.fit_transform(trainIntro)



# TF-IDF:
# Word Level TF-IDFs
title_tfidf_vec = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}')
abstract_tfidf_vec = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}')
intro_tfidf_vec = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}')
title_tfidf_vec.fit(trainTitle)
train_title_tfidf = title_tfidf_vec.transform(trainTitle)
valid_title_tfidf = title_tfidf_vec.transform(validTitle)
abstract_tfidf_vec.fit(trainAbs)
train_abs_tfidf = abstract_tfidf_vec.transform(trainAbs)
valid_abs_tfidf = abstract_tfidf_vec.transform(validAbs)
intro_tfidf_vec.fit_transform(trainIntro)
train_intro_tfidf = intro_tfidf_vec.transform(trainIntro)
valid_intro_tfidf = intro_tfidf_vec.transform(validIntro)
test_intro_tfidf = intro_tfidf_vec.transform(testIntro)


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
print("NB: Count:")
accuracyTitle = train_model(MultinomialNB(), trainTitle_count, train_encodedLabels, validTitle_count)
print("NB, Title Count Vectors: ", accuracyTitle)
accuracyAbs = train_model(MultinomialNB(), train_abs_count, train_encodedLabels, valid_abs_count)
print("NB, Abstract Count Vectors: ", accuracyAbs, "\n")

# NB Word Level TFIDF
print("NB: TFIDF:\n")
accuracyIntro = train_model(MultinomialNB(), train_intro_tfidf, train_encodedLabels, valid_intro_tfidf)
accuracyAbs = train_model(MultinomialNB(), train_abs_tfidf, train_encodedLabels, valid_abs_tfidf)
accuracyTitle = train_model(MultinomialNB(), train_title_tfidf, train_encodedLabels, valid_title_tfidf)
print("NB, Intro WordLevel TF-IDF: ", accuracyIntro)
print("NB, Abstract WordLevel TF-IDF: ", accuracyAbs)
print("NB, Title WordLevel TF-IDF: ", accuracyTitle, "\n")


# Logistic Regression Linear Classification:
# LR Count Vector
print("LR: Count:")
accuracy = train_model(LogisticRegression(), train_abs_count, train_encodedLabels, valid_abs_count)
print("LR, Abstract Count Vectors: ", accuracy, "\n")

# LR Word Level TFIDF
print("LR: Word TFIDF:")
accuracyTitle = train_model(LogisticRegression(), train_title_tfidf, train_encodedLabels, valid_title_tfidf)
print("LR, Title TFIDF: ", accuracyTitle)
accuracyAbstract = train_model(LogisticRegression(), train_abs_tfidf, train_encodedLabels, valid_abs_tfidf)
print("LR, Abstract TFIDF: ", accuracyAbstract)
accuracyIntro = train_model(LogisticRegression(), train_intro_tfidf, train_encodedLabels, valid_intro_tfidf)
print("LR, Intro TFIDF: ", accuracyIntro, "\n")



# Classify Test Data for Submission:
def predict_test(classifier, feature_vector_train, label, feature_vector_test, is_neural_net=False):
    # fit the training dataset on the classifier
    classifier.fit(feature_vector_train, label)
    
    # predict the labels on validation dataset
    predictions = classifier.predict(feature_vector_test)
    print(predictions)

    testDict = {
        # "ID": testData['id'],
        "label": predictions
    }
    testDF = pd.DataFrame(testDict)
    testDF.index.name = "ID"
    print(testDF)
    # save ids and predctions in csv
    testDF.to_csv('results.csv', index=True)

predict_test(LogisticRegression(), train_intro_tfidf, train_encodedLabels, test_intro_tfidf)
