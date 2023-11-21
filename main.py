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
from sklearn.ensemble import VotingClassifier


# Data Preparation:
# Load Datasets to Data Frames
trainFilepath = 'data/train.csv'
testFilepath = 'data/test.csv'
trainData = pd.read_csv(trainFilepath)
testData = pd.read_csv(testFilepath)

# Split into Train and Validate Datasets
trainTitle, validTitle, trainAbs, validAbs, trainIntro, validIntro, trainLabel, validLabel = train_test_split(trainData['title'], trainData['abstract'], trainData['introduction'], trainData['label'])
testTitle = testData['title']
testIntro = testData['introduction']
testAbs = testData['abstract']

# Label Encoder
encoder = LabelEncoder()
train_encodedLabels = encoder.fit_transform(trainLabel)
valid_encodedLabels = encoder.fit_transform(validLabel)



# Count Vectors:
# Train: Create and Transform Count vectorizer object 
trainCount_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
trainCount_vect.fit(trainTitle)
trainCount_vect.fit(trainAbs)
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
test_title_tfidf = title_tfidf_vec.transform(testTitle)
abstract_tfidf_vec.fit(trainAbs)
train_abs_tfidf = abstract_tfidf_vec.transform(trainAbs)
valid_abs_tfidf = abstract_tfidf_vec.transform(validAbs)
test_abs_tfidf = abstract_tfidf_vec.transform(testAbs)
intro_tfidf_vec.fit(trainIntro)
train_intro_tfidf = intro_tfidf_vec.transform(trainIntro)
valid_intro_tfidf = intro_tfidf_vec.transform(validIntro)
test_intro_tfidf = intro_tfidf_vec.transform(testIntro)

# N_Gram Level TF-IDFs
title_tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2,3))
title_tfidf_vect_ngram.fit(trainTitle)
train_title_tfidf_ngram = title_tfidf_vect_ngram.transform(trainTitle)
valid_title_tfidf_ngram = title_tfidf_vect_ngram.transform(validTitle)
test_title_tfidf_ngram = title_tfidf_vect_ngram.transform(testTitle)

intro_tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2,3))
intro_tfidf_vect_ngram.fit(trainIntro)
train_intro_tfidf_ngram = intro_tfidf_vect_ngram.transform(trainIntro)
valid_intro_tfidf_ngram = intro_tfidf_vect_ngram.transform(validIntro)
test_intro_tfidf_ngram = intro_tfidf_vect_ngram.transform(testIntro)

abs_tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2,3))
abs_tfidf_vect_ngram.fit(trainAbs)
train_abs_tfidf_ngram = abs_tfidf_vect_ngram.transform(trainAbs)
valid_abs_tfidf_ngram = abs_tfidf_vect_ngram.transform(validAbs)
test_abs_tfidf_ngram = abs_tfidf_vect_ngram.transform(testAbs)


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
print("NB: TFIDF:")
accuracyIntro = train_model(MultinomialNB(), train_intro_tfidf, train_encodedLabels, valid_intro_tfidf)
accuracyAbs = train_model(MultinomialNB(), train_abs_tfidf, train_encodedLabels, valid_abs_tfidf)
accuracyTitle = train_model(MultinomialNB(), train_title_tfidf, train_encodedLabels, valid_title_tfidf)
print("NB, Intro WordLevel TF-IDF: ", accuracyIntro)
print("NB, Abstract WordLevel TF-IDF: ", accuracyAbs)
print("NB, Title WordLevel TF-IDF: ", accuracyTitle, "\n")

# NB N-Gram TFIDF
print("NB: N-Gram TFIDF")
accuracyTitle = train_model(MultinomialNB(), train_title_tfidf_ngram, train_encodedLabels, valid_title_tfidf_ngram)
print("NB, Title N-Gram TF-IDF: ", accuracyIntro)
accuracyIntro = train_model(MultinomialNB(), train_intro_tfidf_ngram, train_encodedLabels, valid_intro_tfidf_ngram)
print("NB, Intro N-Gram TF-IDF: ", accuracyIntro)
accuracyAbs = train_model(MultinomialNB(), train_abs_tfidf_ngram, train_encodedLabels, valid_abs_tfidf_ngram)
print("NB, Abstract N-Gram TF-IDF: ", accuracyAbs, "\n")



# Logistic Regression Linear Classification:
# LR Count Vector
# print("LR: Count:")
# accuracy = train_model(LogisticRegression(), train_abs_count, train_encodedLabels, valid_abs_count)
# print("LR, Abstract Count Vectors: ", accuracy, "\n")

# LR Word Level TFIDF
print("LR: Word TFIDF:")
accuracyTitle = train_model(LogisticRegression(), train_title_tfidf, train_encodedLabels, valid_title_tfidf)
print("LR, Title TFIDF: ", accuracyTitle)
accuracyAbstract = train_model(LogisticRegression(), train_abs_tfidf, train_encodedLabels, valid_abs_tfidf)
print("LR, Abstract TFIDF: ", accuracyAbstract)
accuracyIntro = train_model(LogisticRegression(), train_intro_tfidf, train_encodedLabels, valid_intro_tfidf)
print("LR, Intro TFIDF: ", accuracyIntro, "\n")

# LR N-Gram TFIDF
print("LR: N-Gram TFIDF")
accuracyTitle = train_model(LogisticRegression(), train_title_tfidf_ngram, train_encodedLabels, valid_title_tfidf_ngram)
print("LR, Title N-Gram TF-IDF: ", accuracyTitle)
accuracyIntro = train_model(LogisticRegression(), train_intro_tfidf_ngram, train_encodedLabels, valid_intro_tfidf_ngram)
print("LR, Intro N-Gram TF-IDF: ", accuracyIntro)
accuracyAbs = train_model(LogisticRegression(), train_abs_tfidf_ngram, train_encodedLabels, valid_abs_tfidf_ngram)
print("LR, Abstract N-Gram TF-IDF: ", accuracyAbs)



# Classify Test Data for Submission:
def predict_test(classifier, feature_vector_train, label, feature_vector_test, is_neural_net=False):
    # fit the training dataset on the classifier
    classifier.fit(feature_vector_train, label)
    
    # predict the labels on validation dataset
    predictions = classifier.predict(feature_vector_test)
    # print(predictions)
    print(classifier.get_params)

    testDict = {
        "ID": testData['ID'],
        "label": predictions
    }
    testDF = pd.DataFrame(testDict)
    testDF.index.name = "ID"
    # print(testDF)
    # save ids and predctions in csv
    testDF.to_csv('results.csv', index=False)

predict_test(LogisticRegression(), train_intro_tfidf, train_encodedLabels, test_intro_tfidf)


# Ensemble Learning Testing:
ensemble = VotingClassifier(estimators=[('NB', MultinomialNB()), ('LR', LogisticRegression())], voting='soft', weights=[0.7, 1])
print("Ensemble: ", ensemble.fit(train_intro_tfidf, train_encodedLabels).score(valid_intro_tfidf, valid_encodedLabels))
