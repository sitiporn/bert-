import os

os.environ['http_proxy'] = 'http://192.41.170.23:3128'
os.environ['https_proxy'] = 'http://192.41.170.23:3128'

import torchtext
import torch
from torch import nn
import math
import numpy as np
import torch.optim as optim
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = 'cuda:1'
print(device)



from torchtext.datasets import IMDB

#training hyperparameters
batch_size = 4
num_epochs = 2
lr=0.0001


from transformers import BertTokenizer, BertForSequenceClassification
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)



train_iter = IMDB(split='train')
test_iter = IMDB(split='test')


trainloader = torch.utils.data.DataLoader(train_iter, batch_size=batch_size)
test_loader = torch.utils.data.DataLoader(test_iter, batch_size=batch_size)


label_maps = {"neg":0,"pos":1}

train_loss = []
test_loss = []


# create optimzer
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
model = model.to(device)

for idx, batch in enumerate(trainloader):
    

    optimizer.zero_grad()
    labels, sample = batch
    #print("labels string  :",labels)
    
    
     
    inputs = tokenizer(sample,padding=True,truncation=True,return_tensors="pt")

    # move parameter to device
    inputs = {k:v.to(device) for k,v in inputs.items()}

    
    labels = [label_maps[stringtoId] for stringtoId in labels]
    
    labels = torch.tensor(labels).unsqueeze(0)


    labels = labels.to(device)
    #print("label Id :",labels)
    
   
    
    outputs = model(**inputs,labels=labels)
    loss, logits = outputs[:2]


     

    loss.backward()
    optimizer.step() 
    
    print("loss :" ,loss.item())
    
    train_loss.append(loss)

    break


     


    #print(loss)
    
    


