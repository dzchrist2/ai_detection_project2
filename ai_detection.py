import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
import sklearn.naive_bayes
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC

# Data Preparation:
# Load Datasets to Data Frames
trainFilepath = 'data/train.csv'
testFilepath = 'data/test.csv'
trainData = pd.read_csv(trainFilepath)
testData = pd.read_csv(testFilepath)

# Label Encoder
encoder = LabelEncoder()
trainLabels = trainData['label']
train_encodedLabels = encoder.fit_transform(trainLabels)

# Feature Extraction:
# Count
intro_count_vec = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
intro_count_vec.fit(trainData['introduction'])
train_intro_count = intro_count_vec.transform(trainData['introduction'])
test_intro_count = intro_count_vec.transform(testData['introduction'])
# TF-IDF
intro_tfidf_vec = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}')
intro_tfidf_vec.fit(trainData['introduction'])
train_intro_tfidf = intro_tfidf_vec.transform(trainData['introduction'])
test_intro_tfidf = intro_tfidf_vec.transform(testData['introduction'])
# Combine
train_features = np.hstack((train_intro_count.toarray(), train_intro_tfidf.toarray()))
test_features = np.hstack((test_intro_count.toarray(), test_intro_tfidf.toarray()))


# Classify Test Data for Submission:
def predict_test(classifier, feature_vector_train, label, feature_vector_test, is_neural_net=False):
    # fit the training dataset on the classifier
    classifier.fit(feature_vector_train, label)
    
    # predict the labels on validation dataset
    predictions = classifier.predict(feature_vector_test)
    # print(predictions)
    # print(classifier.get_params)

    testDict = {
        "ID": testData['ID'],
        "label": predictions
    }
    testDF = pd.DataFrame(testDict)
    testDF.index.name = "ID"
    # print(testDF)
    # save ids and predctions in csv
    testDF.to_csv('results.csv', index=False)

# Ensemble Classifier
ensemble = VotingClassifier(estimators=[('SVC', SVC(probability=True)), ('LR', LogisticRegression())], voting='soft', weights=[1, 0.9])
predict_test(ensemble, train_features, train_encodedLabels, test_features)