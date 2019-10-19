import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer

# Reading csv file data set
mails = pd.read_csv('spam.csv', encoding='utf-8')
mails.head()

# delet blank cells
mails.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)

# rename titles
mails.rename(columns={'v1': 'labels', 'v2': 'message'}, inplace=True)

# Count numbers of spam or ham mails
size = mails['labels'].value_counts()
total = size[0] + size[1]

# Set label spam to 1 and ham to 0 as concept
mails['concept'] = mails['labels'].map({'ham': 0, 'spam': 1})
mails.drop(['labels'], axis=1, inplace=True)

train_index, test_index = list(), list()
for i in range(mails.shape[0]):
    if np.random.uniform(0, 1) < 0.8:
        train_index += [i]
    else:
        test_index += [i]
train_data = mails.loc[train_index]
test_data = mails.loc[test_index]

# rearrange data
train_data.reset_index(inplace=True)
train_data.drop(['index'], axis=1, inplace=True)
test_data.reset_index(inplace=True)
test_data.drop(['index'], axis=1, inplace=True)

message_train = train_data.message
message_test = test_data.message
concept_train = train_data.concept
concept_test = test_data.concept


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
        cl = 'spam'
    else:
        cl = 'ham'
    print('The message tends to be', cl)

    return cl
