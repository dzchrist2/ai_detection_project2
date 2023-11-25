import pandas as pd
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

### Data Preparation: ###
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
trainAll = trainTitle + trainAbs + trainIntro
validAll = validTitle + validAbs + validIntro
testAll = testTitle + testAbs + testIntro

# Label Encoder
encoder = LabelEncoder()
train_encodedLabels = encoder.fit_transform(trainLabel)
valid_encodedLabels = encoder.fit_transform(validLabel)
all_encodedLabels = encoder.fit_transform(trainData['label'])



### Feature Extraction: ###
# Count Vectors:
# Train: Create and Transform Count vectorizer object 
title_count_vec = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
abs_count_vec = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
intro_count_vec = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
all_count_vec = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')

title_count_vec.fit(trainTitle)
train_title_count = title_count_vec.transform(trainTitle)
valid_title_count = title_count_vec.transform(validTitle)
test_title_count = title_count_vec.transform(testTitle)

abs_count_vec.fit(trainAbs)
train_abs_count = abs_count_vec.transform(trainAbs)
valid_abs_count = abs_count_vec.transform(validAbs)
test_abs_count = abs_count_vec.transform(testAbs)

intro_count_vec.fit(trainIntro)
train_intro_count = intro_count_vec.transform(trainIntro)
valid_intro_count = intro_count_vec.transform(validIntro)
test_intro_count = intro_count_vec.transform(testIntro)

all_count_vec.fit(trainAll)
train_all_count = all_count_vec.transform(trainAll)
valid_all_count = all_count_vec.transform(validAll)
test_all_count = all_count_vec.transform(testAll)

# TF-IDF:
# Word Level TF-IDFs
title_tfidf_vec = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}')
abstract_tfidf_vec = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}')
intro_tfidf_vec = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}')
all_tfidf_vec = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}')

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

all_tfidf_vec.fit(trainAll)
train_all_tfidf = all_tfidf_vec.transform(trainAll)
valid_all_tfidf = all_tfidf_vec.transform(validAll)
test_all_tfidf = all_tfidf_vec.transform(testAll)

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

all_tfidf_vec_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2,3))
all_tfidf_vec_ngram.fit(trainAll)
train_all_tfidf_ngram = all_tfidf_vec_ngram.transform(trainAll)
valid_all_tfidf_ngram = all_tfidf_vec_ngram.transform(validAll)


# TODO: add character level ?


# Function: Train Model
def train_model(classifier, feature_vector_train, label, feature_vector_valid):
    # fit the training dataset on classifier
    classifier.fit(feature_vector_train, label)
    # predict the labels on validation dataset
    predictions = classifier.predict(feature_vector_valid)
    # return f1 and accuracy scores
    return [f1_score(predictions, valid_encodedLabels), accuracy_score(predictions, valid_encodedLabels)]


### Classifiers: ###
# Naive Bayes Classification:
# NB Count Vector
print("NB: Count:")
accuracyTitle = train_model(MultinomialNB(), train_title_count, train_encodedLabels, valid_title_count)
print("NB, Title Count Vectors: ", accuracyTitle)
accuracyAbs = train_model(MultinomialNB(), train_abs_count, train_encodedLabels, valid_abs_count)
print("NB, Abstract Count Vectors: ", accuracyAbs)
accuracyIntro = train_model(MultinomialNB(), train_intro_count, train_encodedLabels, valid_intro_count)
print("NB, Intro Count Vectors: ", accuracyIntro)
accuracyAll = train_model(MultinomialNB(), train_all_count, train_encodedLabels, valid_all_count)
print("NB, All Count Vectors: ", accuracyAll, "\n")

# NB Word Level TFIDF
print("NB: TFIDF:")
accuracyIntro = train_model(MultinomialNB(), train_intro_tfidf, train_encodedLabels, valid_intro_tfidf)
accuracyAbs = train_model(MultinomialNB(), train_abs_tfidf, train_encodedLabels, valid_abs_tfidf)
accuracyTitle = train_model(MultinomialNB(), train_title_tfidf, train_encodedLabels, valid_title_tfidf)
accuracyAll = train_model(MultinomialNB(), train_all_tfidf, train_encodedLabels, valid_all_tfidf)
print("NB, Title WordLevel TF-IDF: ", accuracyTitle)
print("NB, Abstract WordLevel TF-IDF: ", accuracyAbs)
print("NB, Intro WordLevel TF-IDF: ", accuracyIntro)
print("NB, ALL TF-IDF: ", accuracyAll, "\n")

# NB N-Gram TFIDF
print("NB: N-Gram TFIDF")
accuracyTitle = train_model(MultinomialNB(), train_title_tfidf_ngram, train_encodedLabels, valid_title_tfidf_ngram)
print("NB, Title N-Gram TF-IDF: ", accuracyIntro)
accuracyAbs = train_model(MultinomialNB(), train_abs_tfidf_ngram, train_encodedLabels, valid_abs_tfidf_ngram)
print("NB, Abstract N-Gram TF-IDF: ", accuracyAbs)
accuracyIntro = train_model(MultinomialNB(), train_intro_tfidf_ngram, train_encodedLabels, valid_intro_tfidf_ngram)
print("NB, Intro N-Gram TF-IDF: ", accuracyIntro)
accuracyAll = train_model(MultinomialNB(), train_all_tfidf_ngram, train_encodedLabels, valid_all_tfidf_ngram)
print("NB, All N-Gram TF-IDF: ", accuracyAll, "\n")


# Logistic Regression Linear Classification:
# LR Count Vector
print("LR: Count:")
accuracyTitle = train_model(LogisticRegression(), train_title_count, train_encodedLabels, valid_title_count)
print("LR, Title Count Vectors: ", accuracyTitle)
accuracyAbs = train_model(LogisticRegression(), train_abs_count, train_encodedLabels, valid_abs_count)
print("LR, Abstract Count Vectors: ", accuracyAbs)
accuracyIntro = train_model(LogisticRegression(), train_intro_count, train_encodedLabels, valid_intro_count)
print("LR, Intro Count Vectors: ", accuracyIntro)
accuracyAll = train_model(LogisticRegression(), train_all_count, train_encodedLabels, valid_all_count)
print("LR, All Count Vectors: ", accuracyAll, "\n")

# LR Word Level TFIDF
print("LR: Word TFIDF:")
accuracyTitle = train_model(LogisticRegression(), train_title_tfidf, train_encodedLabels, valid_title_tfidf)
print("LR, Title TFIDF: ", accuracyTitle)
accuracyAbstract = train_model(LogisticRegression(), train_abs_tfidf, train_encodedLabels, valid_abs_tfidf)
print("LR, Abstract TFIDF: ", accuracyAbstract)
accuracyIntro = train_model(LogisticRegression(), train_intro_tfidf, train_encodedLabels, valid_intro_tfidf)
print("LR, Intro TFIDF: ", accuracyIntro)
accuracyAll = train_model(LogisticRegression(), train_all_tfidf, train_encodedLabels, valid_all_tfidf)
print("LR, ALL TF-IDF: ", accuracyAll, "\n")

# LR N-Gram TFIDF
print("LR: N-Gram TFIDF")
accuracyTitle = train_model(LogisticRegression(), train_title_tfidf_ngram, train_encodedLabels, valid_title_tfidf_ngram)
print("LR, Title N-Gram TF-IDF: ", accuracyTitle)
accuracyAbs = train_model(LogisticRegression(), train_abs_tfidf_ngram, train_encodedLabels, valid_abs_tfidf_ngram)
print("LR, Abstract N-Gram TF-IDF: ", accuracyAbs)
accuracyIntro = train_model(LogisticRegression(), train_intro_tfidf_ngram, train_encodedLabels, valid_intro_tfidf_ngram)
print("LR, Intro N-Gram TF-IDF: ", accuracyIntro)
accuracyAll = train_model(LogisticRegression(), train_all_tfidf_ngram, train_encodedLabels, valid_all_tfidf_ngram)
print("LR, All N-Gram TF-IDF: ", accuracyIntro, "\n")


# Support Vector Classification:
# SVC Count Vector
print("SVC Count:")
accuracyTitle = train_model(SVC(), train_title_count, train_encodedLabels, valid_title_count)
print("SVC, Title Count: ", accuracyTitle)
accuracyAbs = train_model(SVC(), train_abs_count, train_encodedLabels, valid_abs_count)
print("SVC, Abstract Count: ", accuracyAbs)
accuracyIntro = train_model(SVC(), train_intro_count, train_encodedLabels, valid_intro_count)
print("SVC, Intro Count: ", accuracyIntro)
accuracyAll = train_model(SVC(), train_all_count, train_encodedLabels, valid_all_count)
print("SVC, All Count: ", accuracyAll, "\n")

# SVC Word Level TF-IDF
print("SVC Word Level TF-IDF:")
accuracyTitle = train_model(SVC(), train_title_tfidf, train_encodedLabels, valid_title_tfidf)
print("SVC, Title TF-IDF: ", accuracyTitle)
accuracyAbs = train_model(SVC(), train_abs_tfidf, train_encodedLabels, valid_abs_tfidf)
print("SVC, Abstract TF-IDF: ", accuracyAbs)
accuracyIntro = train_model(SVC(), train_intro_tfidf, train_encodedLabels, valid_intro_tfidf)
print("SVC, Intro TF-IDF: ", accuracyIntro)
accuracyAll = train_model(SVC(), train_all_tfidf, train_encodedLabels, valid_all_tfidf)
print("SVC, All TF-IDF: ", accuracyAll, "\n")

# SVC N-Gram Level TF-IDF
print("SVC N-Gram Level TF-IDF:")
accuracyTitle = train_model(SVC(), train_title_tfidf_ngram, train_encodedLabels, valid_title_tfidf_ngram)
print("SVC, Title N-Gram TF-IDF: ", accuracyTitle)
accuracyAbs = train_model(SVC(), train_abs_tfidf_ngram, train_encodedLabels, valid_abs_tfidf_ngram)
print("SVC, Abstract N-Gram TF-IDF: ", accuracyAbs)
accuracyIntro = train_model(SVC(), train_intro_tfidf_ngram, train_encodedLabels, valid_intro_tfidf_ngram)
print("SVC, Intro N-Gram TF-IDF: ", accuracyIntro)
accuracyAll = train_model(SVC(), train_all_tfidf_ngram, train_encodedLabels, valid_all_tfidf_ngram)
print("SVC, All N-Gram TF-IDF: ", accuracyAll, "\n")


# Ensemble Learning Testing:
# Ensemble Count Vector
print("Ensemble Count:")
ensemble = VotingClassifier(estimators=[('SVC', SVC(probability=True)), ('LR', LogisticRegression())], voting='soft', weights=[1, 0.9])
accuracyTitle = train_model(ensemble, train_title_count, train_encodedLabels, valid_title_count)
print("Ensemble, Title Count: ", accuracyTitle)
accuracyAbs = train_model(ensemble, train_abs_count, train_encodedLabels, valid_abs_count)
print("Ensemble, Abstract Count: ", accuracyAbs)
accuracyIntro = train_model(ensemble, train_intro_count, train_encodedLabels, valid_intro_count)
print("Ensemble, Intro Count: ", accuracyIntro)
accuracyAll = train_model(ensemble, train_all_count, train_encodedLabels, valid_all_count)
print("Ensemble, All Count: ", accuracyAll, "\n")

# Ensemble World Level TF-IDF
print("Ensemble Word Level TF-IDF:")
accuracyTitle = train_model(ensemble, train_title_tfidf, train_encodedLabels, valid_title_tfidf)
print("Ensemble, Title TF-IDF: ", accuracyTitle)
accuracyAbs = train_model(ensemble, train_abs_tfidf, train_encodedLabels, valid_abs_tfidf)
print("Ensemble, Abstract TF-IDF: ", accuracyAbs)
accuracyIntro = train_model(ensemble, train_intro_tfidf, train_encodedLabels, valid_intro_tfidf)
print("Ensemble, Intro TF-IDF: ", accuracyIntro)
accuracyAll = train_model(ensemble, train_all_tfidf, train_encodedLabels, valid_all_tfidf)
print("Ensemble, All TF-IDF: ", accuracyAll, "\n")

# Ensemble N-Gram level TF-IDF
print("Ensemble N-Gram TF-IDF:")
accuracyTitle = train_model(ensemble, train_title_tfidf_ngram, train_encodedLabels, valid_title_tfidf_ngram)
print("Ensemble, Title N-Gram TF-IDF: ", accuracyTitle)
accuracyAbs = train_model(ensemble, train_abs_tfidf_ngram, train_encodedLabels, valid_abs_tfidf_ngram)
print("Ensemble, Abstract N-Gram TF-IDF: ", accuracyAbs)
accuracyIntro = train_model(ensemble, train_intro_tfidf_ngram, train_encodedLabels, valid_intro_tfidf_ngram)
print("Ensemble, Intro N-Gram TF-IDF: ", accuracyIntro)
accuracyAll = train_model(ensemble, train_all_tfidf_ngram, train_encodedLabels, valid_all_tfidf_ngram)
print("Ensemble, All N-Gram TF-IDF: ", accuracyAll, "\n")