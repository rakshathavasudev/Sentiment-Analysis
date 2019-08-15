#!/usr/bin/env python
# coding: utf-8

# In[5]:


import nltk,sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import nltk.data
import csv
from nltk.stem.porter import *
pd.options.mode.chained_assignment = None
from sklearn.preprocessing import LabelEncoder
from numpy import array as arr
from numpy import argmax


# In[6]:


arr=[]
file = open("downloaded1.csv",'rt')
samples=csv.reader(file)
c=0
for i in samples:
    c+=1
    
    if c==2:
        x=i[1]
        break

for i in samples:
    if i[1]!=x:
        arr.append(i)


df=pd.DataFrame(data=arr,columns=("types","posts"))
print(len(df.columns))
print(df)


# In[7]:


def labelencode(df):
    data=df['types']
    values=np.array(data)
    label=LabelEncoder()
    intencode=label.fit_transform(values)
    df['typeint']=intencode
    print(list(label.inverse_transform([0,1,2])))
    #df['typeint'].plot(kind='hist')
    #k=np.arange(0,16)
    #x=label.inverse_transform(k)   #can access encoded actual value using x
    #print(values)
    return df

df=labelencode(df)
#print(df)
shortdata=df.iloc[:,1]
#print(shortdata)


# In[8]:


#removing stopwords 
from nltk.corpus import stopwords
stop=stopwords.words("english")
print('------Removing stopwords------')
shortdata=shortdata.apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
#shortdata=shortdata.apply(lambda x: ' '.join([word for word in x.split() if word!='i' or word!='I']))
print(shortdata)
#stemming of words
ps = PorterStemmer()
print('-------Stemming--------')
shortdata = shortdata.apply(lambda x: ' '.join([ps.stem(word) for word in x.split() ]))
print(shortdata)


# In[9]:


shortdata=shortdata.apply(lambda x: ' '.join([word for word in x.split() if word.isalpha()]))

from nltk.stem.wordnet import WordNetLemmatizer
lmtzr = WordNetLemmatizer()
#print(shortdata)
print('-------Lemmatization--------')
shortdata = shortdata.apply(lambda x: ' '.join([lmtzr.lemmatize(word,'v') for word in x.split() ]))
print(shortdata)

print('--------Removing punctuations--------')
def clear_punctuation(s):
	import string
	#print("\n")
	clear_string = ""
	for symbol in s:
		if symbol not in string.punctuation:
			clear_string += symbol
	return clear_string

shortdata = shortdata.apply(lambda x: ''.join(clear_punctuation(x))  )
print(shortdata)


# In[10]:


def strip_all_entities(text):
	import string
	entity_prefixes = ['@']
	for separator in  string.punctuation:
		if separator not in entity_prefixes :
			text = text.replace(separator,' ')
	words = []
	for word in text.split():
		word = word.strip()
		if word:
			if word[0] not in entity_prefixes:
				words.append(word)
	return ' '.join(words)

shortdata = shortdata.apply(lambda x: ''.join(strip_all_entities(x))  ) 


# In[11]:


i=0
arr=[]
print("-----PREPROCESSED_DATA------")
count=0
for line in shortdata:
    df.iloc[i,1]=line
    i=i+1
print(df)


# In[12]:


proc_data=np.array(df['posts'])
label=np.array(df['typeint'])
print(len(proc_data))
print(len(label))


# In[13]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(proc_data, label,
                                                    stratify=label, 
                                                    test_size=0.1)
print(len(X_train))
print(len(X_test))
print(len(y_train))
print(len(y_test))
#print(y_test)


# In[14]:


stoplist = set('for a of the and to in'.split(' '))
texts = [[word for word in document.lower().split() if word not in stoplist] for document in X_train]
#print(texts)
print(len(X_train))
print(X_train)


# In[15]:


from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer(max_features=1000)
X_train_counts = count_vect.fit(X_train)
bowTrain = X_train_counts.transform(X_train)
bowTest = X_train_counts.transform(X_test)
print(bowTrain)
print(bowTest)


# In[20]:


from sklearn.neighbors import KNeighborsClassifier
model=KNeighborsClassifier(n_neighbors=20)
#print(train.shape,trainlabel.shape)
model.fit(bowTrain,y_train)
x=model.predict(bowTest)
print(model.score(bowTrain,y_train))


# In[21]:


count=0
for i in range(len(y_test)):
    if y_test[i]==x[i]:
        count=count+1
print('accuracy:',count/len(y_test)*100)
    


# In[18]:


error = []

# Calculating error for K values between 1 and 40
for i in range(1, 40):  
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(bowTrain, y_train)
    pred_i = knn.predict(bowTest)
    error.append(np.mean(pred_i != y_test))


# In[19]:


plt.figure(figsize=(12, 6))  
plt.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o',  
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')  
plt.xlabel('K Value')  
plt.ylabel('Mean Error')  

