from itertools import chain
import pandas as pd
from pprint import pprint

# Import the data

data = pd.read_csv("data.csv")

# Tokenize the features 

feature_list = list(data["features"])
tokenized_features_list = []

for feature in feature_list:
    tokenized_features_list.append(feature.split())

tokenized_features_list = list(chain.from_iterable(tokenized_features_list))

## Tokenized items have to be unique

tokenized_features_list = list(set(tokenized_features_list))

tokenized_features = {}
for i in range(len(tokenized_features_list)):
    tokenized_features[i] = tokenized_features_list[i]

pprint(tokenized_features)