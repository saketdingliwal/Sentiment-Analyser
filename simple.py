import torch
import json
import sys
import numpy as np
import pickle
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import nltk.sentiment
import random

tokenizer = RegexpTokenizer(r'\w+')
en_stop = set([ "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "it", "it's", "its", "itself", "let's", "me", "more", "most", "my", "myself", "nor", "of", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "she'd", "she'll", "she's", "should", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "we", "we'd", "we'll", "we're", "we've", "were", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "would", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves" ])



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


dirpath = '/home/aman/Desktop/saket/data/'
filepath = dirpath + 'audio_train.json'
documents = []
class_labels = []
summary = []
with open(filepath,'r') as fp:
    line = fp.readline()
    while line:
        input_data = (json.loads(line))
        documents.append(input_data["reviewText"]+input_data["summary"]+input_data["summary"])
        summary.append(input_data["summary"])
        class_label = float(input_data["overall"])
        class_labels.append(class_label)
        line = fp.readline()

class LSTM_MODEL(torch.nn.Module) :
    def __init__(self,vocabsize,embedding_dim,hidden_dim,num_layers,drop_layer,num_classes):
        super(LSTM_MODEL,self).__init__()
        self.hidden_dim = hidden_dim
        self.embeddings = nn.Embedding(vocabsize, embedding_dim)
        self.num_layers = num_layers
        self.lstm = nn.LSTM(embedding_dim,hidden_dim,bidirectional=True)
        self.linearOut = nn.Linear(2*hidden_dim,num_classes)
    def forward(self,inputs):
        x = self.embeddings(inputs).view(len(inputs),batch_size,-1)
        hidden = self.init_hidden()
        lstm_out,lstm_h = self.lstm(x,hidden)
        x = lstm_out[-1]
        x = self.linearOut(x)
        x = F.log_softmax(x)
        return x
    def init_hidden(self):
        h0 = Variable(torch.zeros(2*self.num_layers,batch_size,self.hidden_dim).cuda())
        c0 = Variable(torch.zeros(2*self.num_layers,batch_size,self.hidden_dim).cuda())
        return (h0,c0)


word_count = {}
word_to_idx = {}
counter = 0
def add_word(word):
    global word_to_idx,word_count,counter
    if not word in word_count:
        word_count[word] = 1
    else:
        word_count[word] += 1

def add_doc(document):
    words = document.split()
    for word in words:
        add_word(word)

def limitDict(limit):
    dict1 = sorted(word_count.items(),key = lambda t : t[1], reverse = True)
    count = 0
    for x,y in dict1 :
        if count >= (limit-1) :
            word_to_idx[x] = limit
        else :
            word_to_idx[x] = count
        count+=1

def limitDict2():
    count = 0
    for key in word_count.keys() :
        if word_count[key] >= 6 :
            word_to_idx[key] = count
            count += 1
    word_to_idx['unch'] = count
    count += 1
    word_to_idx['padd'] = count
    count += 1

def clip_doc(doc):
    sent_vect = []
    words = doc.split()
    for word in words:
        if word not in word_to_idx:
            sent_vect.append(word_to_idx['unch'])
        else:
            sent_vect.append(word_to_idx[word])
    if len(sent_vect) > sen_len:
        sent_vect = sent_vect[0:sen_len]
    else:
        diff = sen_len - len(sent_vect)
        for i in range(diff):
            sent_vect.append(word_to_idx['padd'])
    return sent_vect



def vectorize(document):
    global word_to_idx
    input_data = [word_to_idx[word] for word in document.split()]
    return input_data

def batchify(label_doc,start_index):
    label_batch = []
    doc_batch = []
    for i in range(start_index,start_index+batch_size):
        label_batch.append(int(label_doc[i][1])-1)
        doc_batch.append(label_doc[i][0])
    return (label_batch,doc_batch)


documents = documents
class_labels = class_labels
documents = cleaning2(documents)

for document in documents:
    add_doc(document)

limitDict2()
vocabsize = len(word_to_idx)
sen_len = 300
print (vocabsize)

doc_label_pair = []
ind = 0
batch_size = 1000

num_batch = len(documents)//batch_size
# documents = documents[0:num_batch*batch_size]
# class_labels = class_labels[0:num_batch*batch_size]

for ind in range(len(documents)):
    doc_label_pair.append((clip_doc(documents[ind]),class_labels[ind]))


with open(dirpath + 'dict_simple.pkl','wb') as f :
    pickle.dump(word_to_idx,f)
emebed_dim = 400
hidden_dim = 100
model = LSTM_MODEL(vocabsize,emebed_dim,hidden_dim,1,0.5,5)
model = model.cuda()
loss_function = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
epochs = 4
for i in range(epochs):
    loss_sum = 0
    total_acc = 0.0
    random.shuffle(doc_label_pair)
    for iterr in range(num_batch-1):
        label_batch,batch_data = batchify(doc_label_pair,iterr*batch_size)
        batch_data = Variable(torch.LongTensor(batch_data).cuda())
        target_data = Variable(torch.LongTensor(label_batch).cuda())
        class_pred = model(batch_data.t())
        model.zero_grad()
        loss = loss_function(class_pred,target_data)
        loss_sum += loss.data[0]
        loss.backward()
        optimizer.step()
        print ('epoch :',i, 'iterations :',iterr*batch_size, 'loss :',loss.data[0])
    torch.save(model.state_dict(), dirpath+'simple' + str(i+1)+'.pth')
    print ('loss is',(i+1),(loss_sum/len(documents)))
