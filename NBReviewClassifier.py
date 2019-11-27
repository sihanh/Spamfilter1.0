import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer

reviews = pd.read_csv('Review_limited.csv', encoding='utf-8')
# Count numbers of positive[1] and negative[0] review
size = reviews['Concept'].value_counts()
total = size[0] + size[1]

# split data set into training data and testing data with 4:1
train_index, test_index = list(), list()
for i in range(reviews.shape[0]):
    if np.random.uniform(0, 1) < 0.8:
        train_index += [i]
    else:
        test_index += [i]
train_data = reviews.loc[train_index]
test_data = reviews.loc[test_index]

# rearrange split data
train_data.reset_index(inplace=True)
train_data.drop(['index'], axis=1, inplace=True)
test_data.reset_index(inplace=True)
test_data.drop(['index'], axis=1, inplace=True)

# extract test and concept from dataset
message_train = train_data.text
message_test = test_data.text
concept_train = train_data.Concept
concept_test = test_data.Concept

# Setup vectorizer
vectorizer = CountVectorizer(lowercase=True, stop_words='english')

# Train the NB classifier
message_train_counts = vectorizer.fit_transform(message_train.values)
NB_classifier = MultinomialNB()
targets = concept_train.values
NB_classifier.fit(message_train_counts, targets)

# Predict testing set with trained classifier
prediction_train = NB_classifier.predict(message_train_counts)
accuracy_train = accuracy_score(concept_train, prediction_train)

print('')
print('---------------------------------------')
print('SINGLE FOLD RESULT ', '(', 'Naive Bayes', ')')
print('---------------------------------------')
print('')
print('The accuracy score of NB classifier on train set is: ', accuracy_train)
print('')

message_test_counts = vectorizer.transform(message_test.values)
prediction_test = NB_classifier.predict(message_test_counts)
accuracy_test = accuracy_score(concept_test, prediction_test)
print('The accuracy score of NB classifier on test set is: ', accuracy_test)

message_trained_vocabulary = sorted(vectorizer.vocabulary_.items(), key=lambda item:item[1],reverse=True)

# predict single message input
def nb_predict(message):
    # vectorize message
    v_m = vectorizer.transform([message])
    # prediction
    prediction = NB_classifier.predict(v_m)
    if prediction[0] == 1:
        cl = 'positive'
    else:
        cl = 'negative'
    # print('The review is a', cl, 'review')

    return cl

pr_pos = []
pr_neg = []
for text in reviews[reviews['Concept'] == 1]['text']:
    pr_pos.append(nb_predict(text))

for text in reviews[reviews['Concept'] == 0]['text']:
    pr_neg.append(nb_predict(text))

pr_pos = pd.Series(pr_pos)
pr_neg = pd.Series(pr_neg)

size_1 = pr_pos.value_counts()
tp = size_1[0]
fn = size_1[1]
size_2 = pr_neg.value_counts()
tn = size_2[0]
fp = size_2[1]

specificity = tn/(tn+fp)
sensitivity = tp/(tp+fn)
precision = tp/(tp+fp)
tnv = tn/(tn+fn)