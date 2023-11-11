import pandas as pd
from keras.preprocessing import text, sequence
from keras import layers, models, optimizers
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder



filepath = 'data/train.csv'

train = pd.read_csv(filepath, names=['id', 'title', 'abstract', 'introduction', 'label'])
# print(train)

# create a count vectorizer object 
count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
count_vect.fit(train['title'])
# print(count_vect.vocabulary_)
# print("\n\n")
count_vect.fit(train['abstract'])
# print(count_vect.vocabulary_)
count_vect.fit(train['introduction'])

# Encoding
encoder = LabelEncoder()

encodedLabels = encoder.fit_transform(train['label'])

