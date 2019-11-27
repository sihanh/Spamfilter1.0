import random
from NBReviewClassifier import nb_predict
import pandas as pd

first_n_word = ['write','winning','west','wellness','wellman','weapon','weaning','wedding','weddings','yumm']
best_n_word = ['presley','garland','vendredi', 'spécialiste','litter','narines','madison','shrimp','early','yum']

reviews = pd.read_csv('Review_limited.csv')

def gwa_nb(word_list, word_number):
    n = word_number
    pr_pos = []
    pr_neg = []

    for text in reviews[reviews['Concept'] == 1]['text']:
        for i in range(n):
            r = word_list[random.randint(0,9)]
            text = text + ' ' + r
        pr_pos.append(nb_predict(text))

    for text in reviews[reviews['Concept'] == 0]['text']:
        for i in range(n):
            r = word_list[random.randint(0,9)]
            text = text + ' ' + r
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
    accuracy = (tp + tn) / (tp+tn+fp+fn)

    return accuracy, precision, specificity, sensitivity, tnv

n1 = gwa_nb(first_n_word,10)
n2 = gwa_nb(first_n_word,20)
n3 = gwa_nb(first_n_word,50)
n4 = gwa_nb(first_n_word,100)

n5 = gwa_nb(best_n_word,10)
n6 = gwa_nb(best_n_word,20)
n7 = gwa_nb(best_n_word,50)
n8 = gwa_nb(best_n_word,100)
