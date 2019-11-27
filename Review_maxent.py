import nltk.classify.util
from nltk.metrics import precision, recall, f_measure
import pandas as pd
import numpy as np
from nltk.classify import MaxentClassifier
import collections
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


def process_message(message, lower_case=True, stem=True, stop_words=True, gram=2):
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

#
# # Reading csv file data set
 mails = pd.read_csv('Review_limited.csv', encoding='utf-8')
 mails.drop(['Unnamed: 0'],axis=1, inplace=True)
# Count numbers of spam or ham mails
# size = mails['Concept'].value_counts()
# total = size[0] + size[1]
#
# pos_data = mails.loc[mails['Concept'] == 1]
# neg_data = mails.loc[mails['Concept'] == 0]
# pos_data.copy()
# pos_data.copy()
#
# pos_data.drop(['Concept'], axis=1, inplace=True)
# neg_data.drop(['Concept'], axis=1, inplace=True)
#
# ham_data = np.array(pos_data)
# spam_data = np.array(neg_data)
#
# ham_data_processed = []
# spam_data_processed = []
#
# for i in np.arange(ham_data.size):
#     p = process_message(ham_data[i][0])
#     ham_data_processed = np.append(ham_data_processed, p)
#     print(i)
#
# for i in np.arange(spam_data.size):
#     p = process_message(spam_data[i][0])
#     spam_data_processed = np.append(spam_data_processed, p)
#     print(i)
#
# negfeats = [(word_feats(f), 'positive') for f in word_split(spam_data_processed)]
# posfeats = [(word_feats(f), 'negative') for f in word_split(ham_data_processed)]
# negcutoff = int(len(negfeats) * 4 / 5)
# poscutoff = int(len(posfeats) * 4 / 5)
# trainfeats = negfeats[:negcutoff] + posfeats[:poscutoff]
# testfeats = negfeats[negcutoff:] + posfeats[poscutoff:]
#
# train_data_processed = spam_data_processed.tolist()[:negcutoff] + ham_data_processed.tolist()[:poscutoff]
# test_data_processed = spam_data_processed.tolist()[negcutoff:] + ham_data_processed.tolist()[poscutoff:]

# simply load this part to avoid re-processing from above part
with open('maxent_data.pickle','rb') as f:
    negfeats, posfeats, negcutoff, poscutoff, trainfeats, testfeats, train_data_processed, test_data_processed = pickle.load(f)


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
pos_precision = precision(refsets['positive'], testsets['positive'])
pos_recall = recall(refsets['positive'], testsets['positive'])
pos_fmeasure = f_measure(refsets['positive'], testsets['positive'])
neg_precision = precision(refsets['negative'], testsets['negative'])
neg_recall = recall(refsets['negative'], testsets['negative'])
neg_fmeasure = f_measure(refsets['negative'], testsets['negative'])

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
    prediction = {'pos': 0, 'neg': 0}
    for features, Bvalue in p.items():
        features_dict = {features:Bvalue}
        observed = classifier.classify(features_dict)
        prediction[observed] = prediction[observed]+1
    if prediction['pos'] > prediction['neg']:
        cl = 'pos'
    else:
        cl = 'neg'
    # print('The input message is tend to be', cl)

    return cl

