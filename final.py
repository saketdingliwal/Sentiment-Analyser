
# coding: utf-8

# In[1]:


import json
import sys
import sklearn.datasets
import sklearn.metrics
import sklearn.cross_validation
import sklearn.svm
import sklearn.naive_bayes
import sklearn.neighbors
import numpy as np
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk.sentiment
import pickle
negate_list = ["not","never","no"]


# arguments
args = sys.argv


# In[3]:


file = open("../dataset/positive-words.txt")
positive = file.readlines()
positive_set = set()
for i in range(len(positive)):
    positive_set.add(positive[i][:-2])
    positive_set.add(positive[i][:-2]+"_NEG")
file = open("../dataset/negative-words.txt")
negative = file.readlines()
negative_set = set()
for i in range(len(positive)):
    negative_set.add(positive[i][:-2])
    negative_set.add(positive[i][:-2]+"_NEG")


# In[4]:


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


# In[5]:


# nltk.download('stopwords')
tokenizer = RegexpTokenizer(r'\w+')
en_stop = set(stopwords.words('english'))
p_stemmer = PorterStemmer()
wordnet_lemmatizer = WordNetLemmatizer()
# nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger')
train_pickle = "lemmed_doc.pkl"
test_pickle = "lemmed_dev.pkl"


min_df_value = int(args[1])
max_features = int(args[2])
negate_bool = int(args[3])
adj_train_bool = int(args[4])
clean_bool = int(args[5])

# In[6]:


def measures(class_labels,predicted_class):
    print ("accuracy",sklearn.metrics.accuracy_score(class_labels,predicted_class))
    print ("confusion")
    confusion = sklearn.metrics.confusion_matrix(class_labels,predicted_class)
    print (confusion)
    print ("macro f",sklearn.metrics.f1_score(class_labels,predicted_class,average='macro'))
    print ("micro f",sklearn.metrics.f1_score(class_labels,predicted_class,average='micro'))
    print ("weighted f",sklearn.metrics.f1_score(class_labels,predicted_class,average='weighted'))


# In[7]:


def cleaning(docs,adjective):
    new_docs = []
    for document in docs:
        raw = document.lower()
        raw = raw.replace("<br /><br />", " ")
        tokens = tokenizer.tokenize(raw)
        if adjective:
            pos = nltk.pos_tag(tokens)
            adj_list = [tag[0] for tag in pos if tag[1] == 'JJ']
        stopped_tokens = [token for token in tokens if token not in en_stop]
        stemmed_tokens = [p_stemmer.stem(token) for token in stopped_tokens]
        if adjective:
            stemmed_adj_tokens = [p_stemmer.stem(token) for token in adj_list]
            stemmed_tokens = stemmed_tokens + stemmed_adj_tokens
        documentWords = ' '.join(stemmed_tokens)
        new_docs.append(documentWords)
    return new_docs


# In[8]:


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


# In[9]:


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


# In[10]:


def negate(documents):
    new_documents = []
    for doc in documents:
#         doc = doc.lower()
        words = doc.split()
        new_words = not_clear(words)
        newdocument = ' '.join(new_words)
        new_documents.append(newdocument)
    return new_documents


# In[11]:


filepath = '../dataset/audio_train.json'
documents = []
class_labels = []
summary = []

with open(filepath,'r') as fp:
    line = fp.readline()
    while line:
        input_data = (json.loads(line))
        documents.append(input_data["reviewText"])
        summary.append(input_data["summary"])
        class_label = float(input_data["overall"])
        if class_label==1:
            class_labels.append(-2)
        elif class_label==2:
            class_labels.append(-1)
        elif class_label==3:
            class_labels.append(0)
        elif class_label==4:
            class_labels.append(1)
        else:
            class_labels.append(2)
        line = fp.readline()


# In[12]:


dev_documents = []
dev_labels = []
filepath = '../dataset/audio_dev.json'
dev_summary = []
with open(filepath,'r') as fp:
    line = fp.readline()
    while line:
        input_data = (json.loads(line))
        dev_documents.append(input_data["reviewText"])
        class_label = float(input_data["overall"])
        dev_summary.append(input_data["summary"])
        if class_label==1:
            dev_labels.append(-2)
        elif class_label ==2:
            dev_labels.append(-1)
        elif class_label ==3:
            dev_labels.append(0)
        elif class_label==4:
            dev_labels.append(1)
        else:
            dev_labels.append(2)
        line = fp.readline()


# In[13]:


def under_sample(sample_size,documents,labels,summary):
    counter_1 = 0
    counter_2 = 0
    counter_3 = 0
    undersampled_docs = []
    undersampled_labels = []
    undersampled_summary = []
    i = 0
    for i in range(len(documents)):
        if labels[i]==-1 and counter_1 < sample_size:
            undersampled_docs.append(documents[i])
            undersampled_labels.append(labels[i])
            undersampled_summary.append(summary[i])
            counter_1 += 1
        elif labels[i]==0 and counter_2 < sample_size:
            undersampled_docs.append(documents[i])
            undersampled_labels.append(labels[i])
            undersampled_summary.append(summary[i])
            counter_2 += 1
        elif labels[i]==1 and counter_3 < sample_size:
            undersampled_docs.append(documents[i])
            undersampled_labels.append(labels[i])
            undersampled_summary.append(summary[i])
            counter_3 += 1
        if counter_1 == sample_size and counter_2 == sample_size and counter_3 == sample_size:
            break
    return undersampled_docs,undersampled_labels,undersampled_summary


# In[14]:


def get_particular(label,documents,labels,summary):
    undersampled_docs = []
    undersampled_labels = []
    undersampled_summary = []
    i = 0
    for i in range(len(documents)):
        if labels[i] == label:
            undersampled_docs.append(documents[i] + summary[i] + summary[i] + summary[i])
            undersampled_labels.append(labels[i])
            undersampled_summary.append(summary[i])
    return undersampled_docs,undersampled_labels,undersampled_summary


# In[15]:


# undersampled_docs3,undersampled_labels3,undersampled_summary3 = under_sample(74000,documents,class_labels,summary)

# # undersampled_summary = cleaning(summary,0)
# with open('../pickles/complete_summary.pkl', 'rb') as f:
#     cleaned_summary = pickle.load(f)

# with open('../pickles/undersampled_clean.pkl', 'rb') as f:
#     undersampled_docs3 = pickle.load(f)




# with open('../pickles/complete_train_300000_clean.pkl', 'rb') as f:
#     undersampled_docs1 = pickle.load(f)
# with open('../pickles/complete_train_800000_clean.pkl', 'rb') as f:
#     undersampled_docs2 = pickle.load(f)
# cleaned_documents = undersampled_docs1 + undersampled_docs2
# undersampled_docs1,undersampled_labels1,undersampled_summary1 = get_particular(-1,cleaned_documents,class_labels,summary)
# undersampled_docs3,undersampled_labels3,undersampled_summary3 = get_particular(1,cleaned_documents,class_labels,summary)

# undersampled_docs = undersampled_docs1 + undersampled_docs2 + undersampled_docs2[0:20000] +undersampled_docs3[0:200000]
# undersampled_labels = undersampled_labels1 + undersampled_labels2 + undersampled_docs2[0:20000] + undersampled_labels3[0:200000]

# undersampled_docs = negate(documents)
# undersampled_labels = class_labels




# with open('../pickles/complete_train_300000_clean.pkl', 'rb') as f:
#     undersampled_docs1 = pickle.load(f)
# with open('../pickles/complete_train_800000_clean.pkl', 'rb') as f:
#     undersampled_docs2 = pickle.load(f)
# with open('../dataset/summary.pkl','rb') as f:
#     undersampled_summary = pickle.load(f)



# undersampled_docs1,undersampled_labels1,undersampled_summary1 = get_particular(-1,documents,class_labels,summary)
# undersampled_docs2,undersampled_labels2,undersampled_summary2 = get_particular(0,documents,class_labels,summary)
# undersampled_docs = documents
# print len(undersampled_summary)

# for i in range(len(undersampled_docs)):
#     undersampled_docs[i] = undersampled_docs[i] + summary[i] + summary[i] + summary[i]

# undersampled_docs1,undersampled_labels1,undersampled_summary1 = get_particular(-2,documents,class_labels,summary)
# undersampled_docs2,undersampled_labels2,undersampled_summary2 = get_particular(-1,documents,class_labels,summary)
# undersampled_docs3,undersampled_labels3,undersampled_summary3 = get_particular(0,documents,class_labels,summary)
# undersampled_docs4,undersampled_labels4,undersampled_summary4 = get_particular(1,documents,class_labels,summary)
# undersampled_docs5,undersampled_labels5,undersampled_summary5 = get_particular(2,documents,class_labels,summary)
# print len(undersampled_docs1)
# print len(undersampled_docs2)
# print len(undersampled_docs3)
# print len(undersampled_docs4)
# print len(undersampled_docs5)



# In[56]:


undersampled_docs = documents
undersampled_labels = class_labels

for i in range(len(undersampled_docs)):
    undersampled_docs[i] = undersampled_docs[i] + summary[i] + summary[i] + summary[i]

if negate_bool:
    undersampled_docs = negate(undersampled_docs)
if clean_bool:
    undersampled_docs = cleaning2(undersampled_docs)
if adj_train_bool:
    undersampled_docs = add_adj(undersampled_docs)
# undersampled_labels = class_labels



# undersampled_docs = documents
# undersampled_labels = class_labels



# In[57]:





# In[58]:


# dev_documents = cleaning(dev_documents[0:10000],0)
# dev_documents = dev_documents[0:10000]

# with open('../pickles/complete_dev_clean.pkl', 'rb') as f:
#     udev_documents = pickle.load(f)
# with open('../dataset/dev_summary.pkl', 'rb') as f:
#     udev_summary = pickle.load(f)

udev_documents = dev_documents
for i in range(len(udev_documents)):
    udev_documents[i] = udev_documents[i] + dev_summary[i] + dev_summary[i] + dev_summary[i]
if negate_bool:
    udev_documents = negate(udev_documents)
if clean_bool:
    udev_documents = cleaning2(udev_documents)
if adj_train_bool:
    udev_documents = add_adj(udev_documents)

udev_labels = dev_labels

# undersampled_docs = undersampled_docs + udev_documents
# undersampled_labels = undersampled_labels + udev_labels

# undersampled_docs = undersampled_docs[0:100]
# undersampled_labels = undersampled_labels[0:100]

# In[59]:
from sklearn.pipeline import Pipeline
from sklearn import svm
if max_features:
    ppl = Pipeline([
                  ('ngram', sklearn.feature_extraction.text.TfidfVectorizer(token_pattern=r'\b\w+\b', min_df=min_df_value, ngram_range=(1,2),max_df = 0.8,max_features=6000000,stop_words='english')),
                  ('clf',   svm.LinearSVC(C=0.8,class_weight='balanced'))
          ])
else:
    ppl = Pipeline([
                  ('ngram', sklearn.feature_extraction.text.TfidfVectorizer(token_pattern=r'\b\w+\b', min_df=min_df_value, ngram_range=(1,2),max_df = 0.8,stop_words='english')),
                  ('clf',   svm.LinearSVC(C=0.8,class_weight='balanced'))
          ])
model = ppl.fit(undersampled_docs,undersampled_labels)
predicted_dev_class = model.predict(udev_documents)
# if max_features:
#     bigram_vect = sklearn.feature_extraction.text.TfidfVectorizer(token_pattern=r'\b\w+\b', min_df=min_df_value, ngram_range=(1,2),max_df = 0.8,stop_words='english')
# else:
#     bigram_vect = sklearn.feature_extraction.text.TfidfVectorizer(token_pattern=r'\b\w+\b', min_df=min_df_value, ngram_range=(1,2),max_df = 0.8,max_features=5200000,stop_words='english')

# vect.fit(undersampled_docs)
# unigram_vect = sklearn.feature_extraction.text.TfidfVectorizer(stop_words='english')
# X_train_dtm1 = bigram_vect.fit_transform(undersampled_docs)


# X_train_dtm = bigram_vect.fit_transform(undersampled_docs)


# In[60]:


# vect_dev = sklearn.feature_extraction.text.CountVectorizer()
# vect_dev.fit(dev_documents)
# X_dev_dtm2 = unigram_vect.transform(udev_documents)
# X_dev_dtm = bigram_vect.transform(udev_documents)
# X_dev_dtm = hstack((X_dev_dtm1,X_dev_dtm2))


# In[70]:



# clf = svm.LinearSVC(C=0.8,class_weight='balanced').fit(X_train_dtm, undersampled_labels)


# In[71]:


# predicted_class =clf.predict(X_train_dtm)


# In[72]:


# predicted_dev_class = clf.predict(X_dev_dtm)
# print predicted_dev_class
for i in range(len(predicted_dev_class)):
    if predicted_dev_class[i]==2:
        predicted_dev_class[i] = 1
    if predicted_dev_class[i]==-2:
        predicted_dev_class[i] = -1
for i in range(len(udev_labels)):
    if udev_labels[i]==2:
        udev_labels[i] = 1
    if udev_labels[i]==-2:
        udev_labels[i] = -1

# In[73]:


# print "without advectives"
# print "svm"
# print "train"
# measures(undersampled_labels,predicted_class)
# print "dev"
measures(udev_labels,predicted_dev_class)
f1 = sklearn.metrics.f1_score(udev_labels,predicted_dev_class,average='macro')
# with open('../pickles/final_model' + str(f1) + '.pkl', 'w') as f:
#     pickle.dump(clf,f)


# In[53]:


args_string = str(args[1]) + str(args[2]) + str(args[3]) + str(args[4]) + str(args[5])

with open('../pickles/dev_train_model'+ args_string + str(f1) + '.pkl', 'wb') as f:
    pickle.dump(model,f)

#
#
# # In[54]:
#
#
# with open('../pickles/final_dict' + args_string + str(f1) + '.pkl', 'wb') as f:
#     pickle.dump(bigram_vect,f)


# In[65]:


# print np.shape(X_train_dtm)


# In[90]:


# clf_nb = sklearn.naive_bayes.MultinomialNB()
# clf_nb.fit(X_train_dtm,undersampled_labels)
#
#
# # In[91]:
#
#
# predicted_class =clf_nb.predict(X_train_dtm)
# predicted_dev_class = clf_nb.predict(X_dev_dtm)
#
#
# # In[92]:
#
#
# print "nb"
# print "train"
# measures(undersampled_labels,predicted_class)
# print "dev"
# measures(udev_labels,predicted_dev_class)
