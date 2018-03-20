import torch
import json
import sys
import numpy as np
import pickle
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F


dirpath = '/home/cse/btech/cs1150254/Desktop/assign2/'
filepath = dirpath + 'audio_train.json'
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
    input_data = [word_to_idx[word] for word in document.split()]
    return input_data


with open(dirpath + 'dict.pkl','r') as f :
    word_to_idx = pickle.load(f)

vocabsize = len(word_to_idx)
documents = documents[0:10000]
emebed_dim = 50
hidden_dim = 100
model = LSTM_MODEL(vocabsize,emebed_dim,hidden_dim,5)
loss_function = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
epochs = 4
for i in range(epochs):
    loss_sum = 0
    total_acc = 0.0
    iterr = 0
    for document in documents:
        input_data = vectorize(document.lower())
        input_data = Variable(torch.LongTensor(input_data))
        class_label = class_labels[iterr]
        target_data = Variable(torch.LongTensor([int(class_label-1)]))
        class_pred = model(input_data)
        model.zero_grad()
        loss = loss_function(class_pred,target_data)
        loss_sum += loss.data[0]
        loss.backward()
        optimizer.step()
        if iterr%500 == 0 :
            print 'epoch :',i, 'iterations :',iterr, 'loss :',loss.data[0]
        iterr +=1
    torch.save(model.state_dict(), 'model' + str(i+1)+'.pth')
    print 'loss is',(i+1),(loss_sum/len(documents))
