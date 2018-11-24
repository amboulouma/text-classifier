from itertools import chain
import pandas as pd
from pprint import pprint

# Import the data

data = pd.read_csv("data.csv")

# Parsing the features

## Tokenize the features 

## Generate tokens for worlds

feature_list = list(data["features"])
tokenized_features_list = []

for feature in feature_list:
    tokenized_features_list.append(feature.split())

tokenized_features_list = list(chain.from_iterable(tokenized_features_list))

### Tokenized items have to be unique

tokenized_features_list = list(set(tokenized_features_list))

tokenized_features = {}
for i in range(len(tokenized_features_list)):
    tokenized_features[tokenized_features_list[i]] = i

### Replacing worlds with tokens

tokenized_features_list = []

for feature in feature_list:
    tokenized_feature = []
    for world in feature.split():
        tokenized_feature.append(tokenized_features[world])
    tokenized_features_list.append(tuple(tokenized_feature))

pprint(tokenized_features_list)

