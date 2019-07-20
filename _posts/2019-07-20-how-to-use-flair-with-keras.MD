---
layout: post
title: How to use flair with keras
date: 2019-07-19 13:32:20 +0100
description: 
img: flair.jpg # Add image post (optional)
fig-caption: # Add figcaption (optional)
tags: [keras, flair, word-embeddings, bert, python]
---
Zalando's flair and Keras are both beginner-friendly python libraries with great interfaces.
Keras is based on tensorflow and allows defining neural networks within a few lines of code. 
Flair is a multilingual state-of-the-art nlp library and includes typical preprocessing steps like tokenization or POS tagging. 
This tutorial, however, is limited to Flair's ability to handle Word embeddings. 
Since Flair uses pytorch and keras tensorflow, both libraries cannot be used together without some tweaking.
NLP-applications currently require both pre-trained word embeddings and neural networks to deliver contemporary quality.
This introduction deals with the combination of both techniques under usage of flair and keras to keep things simple.<br>
First we define a pandas dataframe to store a dummy text classification dataset. Class 0 contents animal related text and
class 1 texts dealing with traffic matters.
```python
import pandas as pd
texts = ["He played with his cat in the garden.",
         "They went for a walk with teh dog.",
         "She keeps the brid in a large cage.",
         "The engine was broken so we had to repair the old car.",
         "The truck was late because of a huge traffic jam.",
         "He loved to go for a ride on his motorcycle in the late summer."]
classes = [0,0,0,1,1,1]

dataset = pd.DataFrame()
dataset["text"] = texts
dataset["class"] = classes

```
The output should look like this:<br>

|index|text|	class
|0	|He played with his cat in the garden.|	0
|1	|They went for a walk with teh dog.|	0
|2	|She keeps the brid in a large cage.|	0
|3	|The engine was broken so we had to repair the old car. |	1
|4	|The truck was late because of a huge traffic jam. |	1
|5	|He loved to go for a ride on his motorcycle in the late summer.|	1


The next step is to chose one or multiple embeddings we want to use to transform our textdata. Flair currently supports gloVe, fastText, ELMo, Bert and its own flair-embedding.
A common appraoch is to combine a static embedding (gloVe, fastText) with a context sensive embedding by stacking them. In this tutorial we will use fastText and Bert together.
If you haven't used flair by now this will take a while, since the embeddings have to be downloaded from flair repository.

```python
from flair.embeddings import WordEmbeddings, StackedEmbeddings, BertEmbeddings

stacked_embedding = StackedEmbeddings([WordEmbeddings('en'), 
				        BertEmbeddings('bert-base-uncased')])
```
After loading we need to somehow embed our sentences. Flair makes this super easy:
```python
from flair.data import Sentence

text = "They went for a walk with the dog."
sentence = flair.data.Sentence(text) # tokenize data and store in flairs inner format
stacked_embedding.embed(sentence) # add the stacked embedding

for token in sentence:
  print(token.embedding)
```
You could now loop through your dataset store the embeddings somewhere and feed them to your keras network later.
But this has two downsides: First it would take quite a whilefor reading and writing, humilates your harddrive and you would have to rerun this process everytime you want to change something like the emebedding type or the maximum size of your text sequences.
In addition the output format will be a pytorch tensor which can't be read by keras. Instead I recomend to create a python generator to feed the network with data as needed and operating relativly fast in-memory. This looks as follows:<br>
```python
def generateTrainingData(batch_size, max_length, num_classes, emb_size):
  
  x_batch = []
  y_batch = []
  while True:
    data = dataset.sample(frac=1)
    for index, row in data.iterrows():
 
        my_sent = row["text"]
        sentence = Sentence(my_sent)
        stacked_embeddings.embed(sentence)
        
        x = []
        for token in sentence:
          x.append(token.embedding.cpu().detach().numpy())
          if len(x) == max_length:
            break
        
        while len(x) < max_length:
          x.append(np.zeros(emb_size))
        
        y = np.zeros(num_classes)
        y[row["class"]] = 1
        
        x_batch.append(x)            
        y_batch.append(y)

        if len(y_batch) == batch_size:
          yield np.array(x_batch), np.array(y_batch)

          x_batch = []
          y_batch = []
```
Don't worry, we'll go through this step by step. The generator takes four arguments:

```python
def generateTrainingData(batch_size, max_length, num_classes, emb_size):
```

|batch_size| Number of training examples running through your network in one batch
|max_length| Maximum number of tokens in a training example
|num_classes| Number of classes in your dataset
|emb_size| Size of the used embedding for padding. If you are uncertain of your embeddings size use the code snippet below and check the output with np.shape()

<br>
```python
x_batch = []
y_batch = []
```
Initialisation of lists to store training batches. It is common in machine learning contexts to refer x to training data and y to corresponding labels.

```python
while True:
```
This is needed to keep our generator running without predefining the number of epochs and used batches.<br>

```python
data = dataset.sample(frac=1)
```
Every training epoch the dataset will be permutet to prevent the network from simply learning the order of classes.<br>
```python
x = []
for token in sentence:
   x.append(token.embedding.cpu().detach().numpy()) # pytorch-tensor to numpy array
      if len(x) == max_length: # stop if maximum token legth is reached
         break
        
while len(x) < max_length: # fill shorter texts with zero-arrays of the embedding size
   x.append(np.zeros(emb_size))
```



