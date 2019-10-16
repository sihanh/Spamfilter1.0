from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from math import log, sqrt
import pandas as pd
import numpy as np
import re

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

# Split training set with 0.8 ratio 1:4
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

# Count size of training and test data
size_train = train_data['concept'].value_counts()
size_test = test_data['concept'].value_counts()

# Visualize the spam words and ham words with wordcloud
spam_words = ''.join(list(mails[mails['concept'] == 1]['message']))
spam_wc = WordCloud(width=500, height=512).generate(spam_words)
plt.figure(figsize=(10, 8), facecolor='w')
plt.imshow(spam_wc)
plt.axis('off')
plt.tight_layout(pad=0)

ham_words =''.join(list(mails[mails['concept'] == 0]['message']))
ham_wc = WordCloud(width=500, height=500).generate(ham_words)
plt.figure(figsize=(10, 8), facecolor='w')
plt.imshow(ham_wc)
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()

# process message
def process_message(message, lower_case = True, stem = True, stop_words = True, gram = 2):
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

