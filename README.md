# Spamfilter1.0
Naive spam filter using sklearn
# Spam Classifier 

## Naive Bayes Spam Filter
* Dependencies: `sklearn` for classifier frame,`numpy`,`pandas` for data processing.

* The classifier training procedure is in `spam_sky.py`,running this script to train the classifier and view the result. 

* After finishing running, use Function `nb_predict(message)`,`return_type = str` in python console to evaluate a given `message` in `type: str` to classify the message. For example,
```python
 In[1]: nb_predict('I will come by your house tonight after dinner, we can hangout somewhere')
        The message tends to be ham
Out[1]: 'ham'
```

* Example 2
  
```python
 In[2]: nb_predict('Your chance to win 500$ prize! Claim today')
        The message tends to be spam
Out[2]: 'spam'
```
## Maximum Entropy spam fiilter(Maxent filter)

* Depedencies: `nltk`, `pandas`, `numpy`, `collections`

* The classifier training procedure is in `spam_maxent_api.py`,running this script to train the classifier and view the result. The parameter `max_iter` will somehow determine the performance of Maxent Classifier and the larger of the `max_iter`, it will take longer time to train. You can view the trace of training during the running of the script.

* Use function `ma_predict(message)`,`return_type = str` in python console to evaluate a given `message` in `type: str` to classify the message.

* The results are kind of same as the those from previous examples
```python
In[4]:  ma_predict('I will come by your house tonight after dinner, we can hangout somewhere')
        The input message is tend to be ham
Out[4]: 'ham'
-----------------------------------------------------------------------------------------------------
In[5]:  ma_predict('Your chance to win 500$ prize! Claim today')
        The input message is tend to be spam
Out[5]: 'spam'
```
## Good Word Attack
* Run  `good_word_attack.py` to implement active good word attack. Two word lists are generated.
    * First-n-word list was obtained using `CountVectorizer.vocabulary_.items()` from `sklearn.feature_extraction.text`, which gives the n most presented words.
    * Best-n-word list was obtained using `show_most_informative_features(20, show='ham')` from `nltk.classify.MaxentClassifier`, which gives the n  words has most impact.

* The script `good_word_attack.py` will automatically run `spam_sky.py` and `spam_maxent_api.py`. Functions from previous sections are also available to use. 

* Function `word_attack_NB(word_list, message)`, `(word_list = best_n_word or first_n_word)` return the minimum numbers of good words needed from ***word_list***  to turn a spam mail into ham mail using Naive Bayes Classifier. The classifier is more vulnerable if fewer words are needed to break through. This method was proposed by **Christopher Meek** in 2005.

* Function `word_attack_MA(word_list, message)` runs similarly for Maxent filter.

* Example

## NB Reiew Classifier
* apply NB classifiers on yelp review
* similar function usage as before
* `NBReviewClassifier.py`
## Maxent Review Classifier
* apply MA classifiers on yelp review
* similar function usage as before
* Run `Review_maxent.py`
* This Maxent Filter takes about 1 hour to train. Time variance apply on different machines.

## Good word review test
* Evaluate different parameters 
* Run `good_word_review_test.py`
* n1, n2, n3 .... n8 to check parameters under different words adding strategy.

```python
 In[3]: word_attack_NB(first_n_word,'Please call our customer service representative on 0800 169 6031 between 10am-9pm as you have WON a guaranteed a £1000 cash or a £5000 prize')
Out[3]: 2383
```
