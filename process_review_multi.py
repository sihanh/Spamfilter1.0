import pandas as pd 
import json
from langdetect import detect
import re
from multiprocessing import Pool
import numpy as np
import time

# load massive data .json file
review = pd.read_json('review.json', lines=True, chunksize = 100000)
chunks = []

# Define stars to flag function
def star_to_flag(number):
    if number > 2:
        return 1 #'Positive'
    else:
        return 0 #'Negative'

def remove_sc(value):
    """
    Eliminate all the non-letter factors from content including numbers, special characters
    :param value: content need to be processed
    :return: return content after processing
    """
    # \W 表示匹配非数字字母下划线
    result = re.sub('\W+', '', value).replace("_", '')
    return result

def text_language_detect(text):
    """
    Detect language of the str
    :parameter text: text need to be detected
    :return: a string representing language such as 'zh-cn','zh-tw','en', etc.
    """
    if remove_sc(text) is not '':
        return detect(remove_sc(text))
    else:
        return 'unknown'
        
def map_tld(df):
    return df.map(text_language_detect, na_action='ignored')

def init_process(global_vars):
    global a
    a = global_vars

if __name__ == "__main__":
    a = 2
    start = time.time()
    #long running
    #do something other

    # processing data in the following for loop
    for chunk in review:
        # apply data cleaning here
        chunk = chunk[chunk['useful']>3] # select useful reviews set 3 as cutoff
        # use parrallel to boost processing
        text_parts = np.array_split(chunk['text'],20) 
        
        with Pool(processes=8, initializer=init_process,initargs=(a,)) as pool:        
            result_parts = pool.map(map_tld, text_parts)

        chunk['Language'] = pd.concat(result_parts, ignore_index =True)
        
        chunk[chunk['Language'] == 'en'] # extract only english reviews
        chunk['Concept'] = chunk['stars'].map(star_to_flag) # more than 2 starts -> positive else negative
        chunk.drop(['review_id','business_id','user_id','cool','funny','date','useful','stars'], axis=1, inplace=True)
        chunks.append(chunk)
        break

    df = pd.concat(chunks,ignore_index=True)
    end = time.time()
    print(end-start)
    df.head()