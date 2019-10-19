import nltk.classify.util
from nltk.metrics import precision, recall, f_measure
import pandas as pd
import numpy as np
from nltk.classify import MaxentClassifier
import collections
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


def process_message(message, lower_case=True, stem=True, stop_words=True, gram=1):
    if lower_case:
        message = message.lower()
    words = word_tokenize(message)
    words = [w for w in words if len(w) > 2]
    if gram > 1:
        w = []
        for i in range(len(words) - gram + 1):
            w += [' '.join(words[i:i + gram])]
        return w
    if stop_words:
        sw = stopwords.words('english')
        words = [word for word in words if word not in sw]
    if stem:
        stemmer = PorterStemmer()
        words = [stemmer.stem(word) for word in words]
    return words


def word_split(data):
    data_new = []
    for word in data:
        word_filter = [i.lower() for i in word.split()]
        data_new.append(word_filter)
    return data_new


def word_feats(words):
    return dict([(word, True) for word in words])

# Reading csv file data set
mails = pd.read_csv('spam.csv', encoding='utf-8')
mails.head()
# delete blank cells
mails.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)
# rename titles
mails.rename(columns={'v1': 'labels', 'v2': 'message'}, inplace=True)
# Count numbers of spam or ham mails
size = mails['labels'].value_counts()
total = size[0] + size[1]
# Set label spam to 1 and ham to 0 as concept
mails['concept'] = mails['labels'].map({'ham': 0, 'spam': 1})
mails.drop(['labels'], axis=1, inplace=True)

pos_data = mails.loc[mails['concept'] == 0]
neg_data = mails.loc[mails['concept'] == 1]
pos_data.copy()
pos_data.copy()

pos_data.drop(['concept'], axis=1, inplace=True)
neg_data.drop(['concept'], axis=1, inplace=True)

ham_data = np.array(pos_data)
spam_data = np.array(neg_data)

ham_data_processed = []
spam_data_processed = []

for i in np.arange(ham_data.size):
    p = process_message(ham_data[i][0])
    ham_data_processed = np.append(ham_data_processed, p)

for i in np.arange(spam_data.size):
    p = process_message(spam_data[i][0])
    spam_data_processed = np.append(spam_data_processed, p)

# classifier training
# def classifier_performance(spam_data_processed, ham_data_processed):

negfeats = [(word_feats(f), 'spam') for f in word_split(spam_data_processed)]
posfeats = [(word_feats(f), 'ham') for f in word_split(ham_data_processed)]
negcutoff = int(len(negfeats) * 4 / 5)
poscutoff = int(len(posfeats) * 4 / 5)
trainfeats = negfeats[:negcutoff] + posfeats[:poscutoff]
testfeats = negfeats[negcutoff:] + posfeats[poscutoff:]

train_data_processed = spam_data_processed.tolist()[:negcutoff] + ham_data_processed.tolist()[:poscutoff]
test_data_processed = spam_data_processed.tolist()[negcutoff:] + ham_data_processed.tolist()[poscutoff:]

# using Maxent classifiers
classifierName = 'Maximum Entropy'
classifier = MaxentClassifier.train(trainfeats, max_iter=50)
refsets = collections.defaultdict(set)
testsets = collections.defaultdict(set)

for i, (feats, label) in enumerate(testfeats):
    refsets[label].add(i)
    observed = classifier.classify(feats)
    testsets[observed].add(i)

accuracy = nltk.classify.util.accuracy(classifier, testfeats)
pos_precision = precision(refsets['ham'], testsets['ham'])
pos_recall = recall(refsets['ham'], testsets['ham'])
pos_fmeasure = f_measure(refsets['ham'], testsets['ham'])
neg_precision = precision(refsets['spam'], testsets['spam'])
neg_recall = recall(refsets['spam'], testsets['spam'])
neg_fmeasure = f_measure(refsets['spam'], testsets['spam'])

print('')
print('---------------------------------------')
print('SINGLE FOLD RESULT ', '(', classifierName, ')')
print('---------------------------------------')
print('accuracy:', accuracy)
# classifier.show_most_informative_features(50, show='ham')
print('')
# classifier.show_most_informative_features(20, show='ham')


# Return dictionary with value show if the words in train data
def predict_feats(words):
    d = dict()
    for word in words:
        if word in train_data_processed:
            d[word] = True
        else:
            d[word] = False
    return d
    # return dict([(word, True) for word in words]) if word in


def ma_predict(message):
    p = predict_feats(process_message(message))
    prediction = {'ham': 0, 'spam': 0}
    for features, Bvalue in p.items():
        features_dict = {features:Bvalue}
        observed = classifier.classify(features_dict)
        prediction[observed] = prediction[observed]+1
    if prediction['ham'] > prediction['spam']:
        cl = 'ham'
    else:
        cl = 'spam'
    # print('The input message is tend to be', cl)

    return cl

