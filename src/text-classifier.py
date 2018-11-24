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
    tokenized_features[tokenized_features_list[i]] = i + 1


### Replacing worlds with tokens

tokenized_features_list = []

for feature in feature_list:
    tokenized_feature = []
    for world in feature.split():
        tokenized_feature.append(tokenized_features[world])
    tokenized_features_list.append(tokenized_feature)


## Padding the vectors so that they will have the same length 

### Having the max length

max_length = len(max(tokenized_features_list, key=len))

for vector in tokenized_features_list:
    while len(vector) < 6:
        vector.append(0)

# Tokenizing labels

labels_list = list(data["labels"])

tokenized_labels = {
    "yes": 1,
    "no": 0,
    "maybe": 0.5
}

tokenized_labels_list = []

for label in labels_list:
    tokenized_labels_list.append(tokenized_labels[label])

pprint(tokenized_features_list)
pprint(tokenized_labels_list)
