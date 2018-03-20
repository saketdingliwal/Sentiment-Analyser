import torch
import json
import sys
import numpy as np
import pickle
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import sklearn.datasets
import sklearn.metrics
import sklearn.cross_validation
import sklearn.svm
import sklearn.naive_bayes
import sklearn.neighbors


dirpath = '/home/cse/btech/cs1150254/Desktop/assign2/'
filepath = dirpath + 'audio_dev.json'
documents = []
class_labels = []
summary = []

def measures(class_labels,predicted_class):
    print ("accuracy",sklearn.metrics.accuracy_score(class_labels,predicted_class))
    print ("confusion")
    confusion = sklearn.metrics.confusion_matrix(class_labels,predicted_class)
    print confusion
    print "macro f",sklearn.metrics.f1_score(class_labels,predicted_class,average='macro')
    print "micro f",sklearn.metrics.f1_score(class_labels,predicted_class,average='micro')
    print "weighted f",sklearn.metrics.f1_score(class_labels,predicted_class,average='weighted')



with open(filepath,'r') as fp:
    line = fp.readline()
    while line:
        input_data = (json.loads(line))
        documents.append(input_data["reviewText"])
        summary.append(input_data["summary"])
        class_label = float(input_data["overall"])
        class_label = int(class_label)-1
        class_labels.append(class_label)
        line = fp.readline()

class LSTM_MODEL(torch.nn.Module) :
    def __init__(self,vocabsize,embedding_dim,hidden_dim,num_classes):
        super(LSTM_MODEL,self).__init__()
        self.hidden_dim = hidden_dim
        self.embeddings = nn.Embedding(vocabsize, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim,hidden_dim)
        self.linearOut = nn.Linear(hidden_dim,num_classes)
    def forward(self,inputs):
        x = self.embeddings(inputs).view(len(inputs),1,-1)
        hidden = self.init_hidden()
        lstm_out,lstm_h = self.lstm(x,hidden)
        x = lstm_out[-1]
        x = self.linearOut(x)
        x = F.log_softmax(x)
        return x
    def init_hidden(self):
        h0 = Variable(torch.zeros(1,1,self.hidden_dim))
        c0 = Variable(torch.zeros(1,1, self.hidden_dim))
        return (h0,c0)





word_to_idx = {}
count = 0
def add_word(word):
    global word_to_idx,count
    if not word in word_to_idx:
        word_to_idx[word] = count
        count +=1
def add_doc(document):
    words = document.split()
    for word in words:
        add_word(word)

# for document in documents:
#     add_doc(document.lower())

def vectorize(document):
    global word_to_idx
    input_data = []
    for word in document.split():
        if word in word_to_idx:
            input_data.append(word_to_idx[word])
    return input_data


with open(dirpath + 'dict.pkl','r') as f :
    word_to_idx = pickle.load(f)

model.load_state_dict(torch.load('model0.pth'))

vocabsize = len(word_to_idx)
documents = documents[0:10000]
class_labels = class_labels[0:10000]
predicted_labels = []
iterr=0
for document in documents:
    input_data = vectorize(document.lower())
    input_data = Variable(torch.LongTensor(input_data))
    y_pred = model(input_data)
    pred1 = y_pred.data.max(1)[1].numpy()
    predicted_labels.append(pred1[0][0])
    iterr+=1
measures(class_labels,predicted_labels)
