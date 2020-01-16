---
layout: post
title: How to use flair with keras
date: 2019-07-19 13:32:20 +0100
description: 
img: flair.jpg # Add image post (optional)
fig-caption: # Add figcaption (optional)
tags: [keras, flair, word-embeddings, bert, python]
---
Zalando's <a href="https://research.zalando.com/welcome/mission/research-projects/flair-nlp/">flair</a> and <a href="https://keras.io/why-use-keras/">Keras</a> are both beginner-friendly python libraries with great interfaces.
Keras is based on tensorflow and allows defining neural networks within a few lines of code. 
Flair is a multilingual state-of-the-art nlp library and includes typical preprocessing steps like tokenization or POS tagging. 
This tutorial, however, is limited to Flair's ability to handle word embeddings. 
Since Flair uses pytorch and keras tensorflow, both libraries cannot be used together without some tweaking.
NLP-applications currently require both pre-trained word embeddings and neural networks to deliver cutting-edge quality.
To keep things simple, this introduction deals with the combination 
of both techniques.<br>
First we define a pandas dataframe to store a dummy text classification dataset. Our task is the classification of short texts; class 0 contains 
texts about animals and class 2 texts dealing with cars and traffic.
```python
import pandas as pd
texts = ["He played with his cat in the garden.",
         "They went for a walk with the dog.",
         "She keeps the bird in a large cage.",
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
|1	|They went for a walk with the dog.|	0
|2	|She keeps the bird in a large cage.|	0
|3	|The engine was broken so we had to repair the old car. |	1
|4	|The truck was late because of a huge traffic jam. |	1
|5	|He loved to go for a ride on his motorcycle in the late summer.|	1


The next step is to choose one or multiple embeddings we want to use to transform our textdata. Flair currently supports gloVe, fastText, ELMo, Bert and its own flair-embedding.
A common appraoch is to combine a static embedding (gloVe, fastText) with a context sensitive embedding by stacking them. In this tutorial we will use fastText and Bert together.
If you haven't used flair before, this may take a while, since the embeddings have to be downloaded from the repository.

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
You could now loop through your dataset, store the embeddings somewhere and feed them to your keras network later.
But this has two downsides: First it would take quite a while for reading and writing, stresses your harddrive and you would have to rerun this process everytime you want to change something like the emebedding type or the maximum size of your text sequences.
In addition the output format will be a pytorch tensor which can't be read by keras. Instead I recomend to create a python generator to feed the network with data as needed and operating relativly fast in-memory. This looks as follows:<br>
```python
def generateTrainingData(dataset, batch_size, max_length, num_classes, emb_size):
  
  x_batch = []
  y_batch = []
  while True:
    data = dataset.sample(frac=1)
    for index, row in data.iterrows():
 
        my_sent = row["text"]
        sentence = Sentence(my_sent)
        stacked_embedding.embed(sentence)
        
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
Don't worry, we'll go through this step by step. The generator takes five arguments:

```python
def generateTrainingData(dataset, batch_size, max_length, num_classes, emb_size):
```

|dataset| Training data stored in a pandas dataframe
|batch_size| Number of training examples running through your network in one batch
|max_length| Maximum number of tokens in a training example
|num_classes| Number of classes in your dataset
|emb_size| Size of the used embedding for padding. If you are uncertain of your embeddings size use the code snippet below and check the output with np.shape()

<br>
```python
x_batch = []
y_batch = []
```
Initialisation of lists to store training batches.In machine learning contexts one usually refers to the training data as x and to the corresponding labels as y.

```python
while True:
```
This is needed to keep our generator running without predefining the number of epochs and used batches.<br>

```python
data = dataset.sample(frac=1)
```
Every training epoch the dataset will be permutetated to prevent the network from simply learning the order of classes.<br>
```python
x = []
for token in sentence:
   x.append(token.embedding.cpu().detach().numpy()) # pytorch-tensor to numpy array
      if len(x) == max_length: # stop if maximum token legth is reached
         break
        
while len(x) < max_length: # fill 
shorter texts with zero-arrays; each array has the same dimension as the 
word embeddings
   x.append(np.zeros(emb_size))
```
<br>
This code block loops over every token in the current text segment. If the segment is longer than the maximum token length it will stop. If this is not the case
the rest will be filled with arrays containing just zeros. To handle the pytorch tensors they are converted to numpy arrays. To make sure the network can use all
the GPU resources available this is done with pytorch in CPU mode. 
```python
y = np.zeros(num_classes)
y[row["class"]] = 1
```
These two lines transform classes to one-hot-encoding.
<br>
```python
if len(y_batch) == batch_size:
   yield np.array(x_batch), np.array(y_batch)

   x_batch = []
   y_batch = []
```
If enough examples are collected to form one batch they are handed to the network. The lists get cleared and the generator starts to create the next batch.

```python
from keras.layers import Input, Dense, GRU, Bidirectional, Flatten
from keras.optimizers import Adam
from keras.models import Model

def declare_model(dataset, batch_size, max_len, emb_size, gru_size, num_classes):
 
  sample = Input(batch_shape=(batch_size, max_len, emb_size))
  gru_out = Bidirectional(GRU(gru_size, return_sequences=True))(sample)
  gru_out = Flatten()(gru_out)
  predictions = Dense(num_classes, activation='sigmoid')(gru_out)

  model = Model(inputs=sample, outputs=[predictions])
  model.compile(optimizer=Adam(),loss='binary_crossentropy', metrics=["acc"])
  print(model.summary())

  return model
m = declare_model(batch_size=2, max_len=10, emb_size=3372, gru_size=20, num_classes=2)
```
The next step is the definition of a keras network. I don't want to go into details because there are already several 
pretty handy tutorials on that topic out there. For example: <a href="https://realpython.com/python-keras-text-classification/">link</a>.
I especially recommend <a href="https://www.manning.com/books/deep-learning-with-python">"Deep Learning with Python"</a> by François Chollet.
<br>
```python
gen = generateTrainingData(data_set=data_set, batch_size=2, max_length=10, num_classes=2, emb_size=3372)
m.fit_generator(gen, steps_per_epoch=3, epochs=10, max_queue_size=10, workers=1)
```
Now we can start to pass batches through the network with the function fit_generator(). In contrast to using just fit() the model does not know the exact size of its trainingdata, therefore it is
needed to set steps_per_epoch to the batch count that equals a loop through the whole dataset. The argument max_queue_size holds the information on how many batches should be generated and hold in memory to provide an optimal flow of training data for the network. The maximum usage of parallel cpu cores is fixed by the workers argument. It can be helpful, especially for long running tasks, to play around with batch_size, max_queue_size and workers to find the optimal parameters balancing GPU and CPU load for the fastest learning process.<br>
But lets see how the network handles unseen sentences after facing just 6 training examples. To perform a prediction a variation of the former generator without yielding labels has to be decalred.
```python
def generatePredictionData(dataset, batch_size, max_length, num_classes, emb_size):
  
  x_batch = []
  while True:
    for text in dataset:
 
        my_sent = text
        sentence = Sentence(my_sent)
        stacked_embedding.embed(sentence)
        
        x = []
        for token in sentence:
          x.append(token.embedding.cpu().detach().numpy())
          if len(x) == max_length:
            break
        
        while len(x) < max_length:
          x.append(np.zeros(emb_size))
          
        x_batch.append(x)            
        if len(x_batch) == batch_size:
          yield np.array(x_batch)

          x_batch = []

test_sents = ["He held his hamster close.",
              "She tried to start but realised there was a marten damage."]

gen = generatePredictionData(dataset=test_sents, batch_size=2, max_length=10, num_classes=2, emb_size=3372)
print(np.argmax(m.predict_generator(gen, steps=1), axis=1))

# output:
# [0 1]

```
Everything works out like expected and the first sentence is classified as animal related. The
network doesn't even misclassify the tricky example about the 'marten damage'.<br> 
I hope you learned something by reading this tutorial. If you have any suggestions on optimizing the code, please let me know.
