
# coding: utf-8

# In[1]:

import sys
import json
import sklearn.datasets
import sklearn.metrics
import sklearn.cross_validation
import sklearn.svm
import sklearn.naive_bayes
import sklearn.neighbors
import numpy as np
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import pickle
negate_list = ["not","never","no"]
tokenizer = RegexpTokenizer(r'\w+')
en_stop = set(stopwords.words('english'))

def add_adj(documents):
    new_docs = []
    for doc in documents:
        doc = doc.lower()
        new_words = []
        words = doc.split()
        for word in words:
            new_words.append(word)
            if word in positive_set or word in negative_set:
                new_words.append(word)
                new_words.append(word)
        new_doc = ' '.join(new_words)
        new_docs.append(new_doc)
    return new_docs



# In[ ]:


test_documents = []
filepath = sys.argv[1]
test_summary = []
with open(filepath,'r') as fp:
    line = fp.readline()
    while line:
        input_data = (json.loads(line))
        test_documents.append(input_data["reviewText"]+input_data["summary"]+input_data["summary"]+input_data["summary"])
        line = fp.readline()


# In[1]:


def cleaning2(docs):
    new_docs = []
    for document in docs:
        raw = document.lower()
        raw = raw.replace("<br /><br />", " ")
        tokens = tokenizer.tokenize(raw)
        stopped_tokens = [token for token in tokens if token not in en_stop]
        documentWords = ' '.join(stopped_tokens)
        new_docs.append(documentWords)
    return new_docs


# In[2]:


def not_clear(tokens):
    i =0
    for token in tokens:
        if token in negate_list or token[-3:]=="n't":
            if i+1 < len(tokens):
                tokens[i+1] =  tokens[i+1] + "_NEG"
            if i+2 < len(tokens):
                tokens[i+2] = tokens[i+2] + "_NEG"
        i+=1
    return tokens


# In[3]:


def negate(documents):
    new_documents = []
    for doc in documents:
#         doc = doc.lower()
        words = doc.split()
        new_words = not_clear(words)
        newdocument = ' '.join(new_words)
        new_documents.append(newdocument)
    return new_documents


# In[ ]:


# with open('clf.pkl', 'rb') as f:
#     clf = pickle.load(f)
# with open('dict.pkl', 'rb') as f:
#     bigram_vect = pickle.load(f)

with open('model.pkl','rb') as f:
    model = pickle.load(f)
# In[ ]:

file = open("positive-words.txt")
positive = file.readlines()
positive_set = set()
for i in range(len(positive)):
    positive_set.add(positive[i][:-2])
    positive_set.add(positive[i][:-2]+"_NEG")
file = open("negative-words.txt")
negative = file.readlines()
negative_set = set()
for i in range(len(positive)):
    negative_set.add(positive[i][:-2])
    negative_set.add(positive[i][:-2]+"_NEG")

# print "cleaning"
utest_docs = negate(test_documents)
utest_docs = cleaning2(utest_docs)
utest_docs = add_adj(utest_docs)
# print "done cleaning"


# In[ ]:

# print "starting prediction"
X_test_dtm = bigram_vect.transform(utest_docs)


# In[ ]:


# predicted_class =clf.predict(X_test_dtm)
predicted_class = model.predict(utest_docs)

# In[ ]:


output_file = open(sys.argv[2],'w')
for i in range(len(predicted_class)):
    if predicted_class[i]<0:
        output_file.write("1\n")
    elif predicted_class[i]==0:
        output_file.write("3\n")
    else:
        output_file.write("5\n")
