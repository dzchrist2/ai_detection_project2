import pandas as pd
from keras.preprocessing import text, sequence
from keras import layers, models, optimizers

filepath = 'data/train.csv'

train = pd.read_csv(filepath, names=['id', 'title', 'abstract', 'introduction', 'label'])

print(train)



