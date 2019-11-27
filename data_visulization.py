import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from nltk.corpus import stopwords

# configure stop words
stop = stopwords.words('english')

# load review data
data = pd.read_csv('Review_limited.csv', encoding='utf-8')
data['text'] = data["text"].str.lower().str.split()
data['text'] = data['text'].apply(lambda x: [item for item in x if item not in stop])
data['text'] = data['text'].apply(' '.join)

# elet unnamed part
data.drop(['Unnamed: 0'], axis=1, inplace=True)

# plot bar figure
# count_Class = pd.value_counts(data["Concept"], sort=True)
# count_Class.plot(kind='bar', color=["blue", "orange"])
# plt.title('Bar chart')
# plt.show()
#
# #plot pie figure
# count_Class.plot(kind = 'pie',  autopct='%1.0f%%')
# plt.title('Pie chart')
# plt.ylabel('')
# plt.show()

# count most common word
count1 = Counter(" ".join(data[data['Concept']==1]['text']).split()).most_common(20)
df1 = pd.DataFrame.from_dict(count1)
df1 = df1.rename(columns={0: "words in positive review", 1: "count"})
count2 = Counter(" ".join(data[data['Concept']==0]['text']).split()).most_common(20)
df2 = pd.DataFrame.from_dict(count2)
df2 = df2.rename(columns={0: "words in negative review", 1: "count_"})

# plot bar code of words distribution
df1.plot.bar(legend = False)
y_pos = np.arange(len(df1["words in positive review"]))
plt.xticks(y_pos, df1["words in positive review"])
plt.title('More frequent words in positive reviews')
plt.xlabel('words')
plt.ylabel('number')
plt.show()

df2.plot.bar(legend = False, color = 'orange')
y_pos = np.arange(len(df2["words in negative review"]))
plt.xticks(y_pos, df2["words in negative review"])
plt.title('More frequent words in negative reviews')
plt.xlabel('words')
plt.ylabel('number')
plt.show()


