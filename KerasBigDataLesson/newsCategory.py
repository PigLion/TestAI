import pandas as pd
import numpy as np
from keras import models
from keras import layers
import matplotlib.pyplot as plt
from keras import optimizers
from keras.preprocessing.text import Tokenizer


amount=60000
newsData = pd.read_json('../../_data/News_Category_Dataset_v2.json', lines=True)
newsData=newsData.drop(columns=['date','link'],axis=1)
newsData= newsData.dropna(how='any')


newsData.category=newsData.category.map(lambda x: "WORLDPOST" if x == "THE WORLDPOST" else x)
newsData['text'] = newsData.headline + " " + newsData.short_description
tokenizer = Tokenizer()
tokenizer.fit_on_texts(newsData.text)
newsData['words'] = tokenizer.texts_to_sequences(newsData.text)
# print(newsData.loc[:100,'words'])

def vectorize_sequences(sequences):
    dimension=10000
    results = np.zeros((amount, dimension))
    for i in range(amount):
        for k in sequences[i]:
            if(k<10000):
                results[i,k] = 1.
    return results

inputData=vectorize_sequences(newsData.words)
print(inputData[:4,:10])
print(newsData.words[:4])
print("--------------------")


model = models.Sequential()
model.add(layers.Dense(64, activation='sigmoid',
                       input_shape=(inputData.shape[1],)))
model.add(layers.Dense(64, activation='sigmoid'))
model.add(layers.Dense(40,activation='sigmoid'))
model.compile(optimizer=optimizers.RMSprop(lr=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print(np.unique(newsData.category))

targetMap={
    'ARTS':0,
    'ARTS & CULTURE':1,
    'BLACK VOICES':2,
    'BUSINESS':3,
    'COLLEGE':4,
    'COMEDY':5,
    'CRIME':6,
    'CULTURE & ARTS':7,
    'DIVORCE':8,
    'EDUCATION':9,
    'ENTERTAINMENT':10,
    'ENVIRONMENT':11,
    'FIFTY':12,
    'FOOD & DRINK':13,
    'GOOD NEWS':14,
    'GREEN':15,
    'HEALTHY LIVING':16,
    'HOME & LIVING':17,
    'IMPACT':18,
    'LATINO VOICES':19,
    'MEDIA':20,
    'MONEY':21,
    'PARENTING':22,
    'PARENTS':23,
    'POLITICS':24,
    'QUEER VOICES':25,
    'RELIGION':26,
    'SCIENCE':27,
    'SPORTS':28,
    'STYLE':29,
    'STYLE & BEAUTY':30,
    'TASTE':31,
    'TECH':32,
    'TRAVEL':33,
    'WEDDINGS':34,
    'WEIRD NEWS':35,
    'WELLNESS':36,
    'WOMEN':37,
    'WORLD NEWS':38,
    'WORLDPOST':39
}

def to_one_hot(sequences):
    dimension=40
    results = np.zeros((amount, dimension))
    for i in range(amount):
        results[i,sequences[i]] = 1.
    return results
target=to_one_hot(newsData.category.map(targetMap))

train_data=inputData[:40000]
test_data=inputData[40000:50000]
valiation_data=inputData[50000:]
train_target=target[:40000]
test_target=target[40000:50000]
validation_target=target[50000:]




num_epochs = 20

history = model.fit(train_data, train_target,
                    validation_data=(valiation_data, validation_target),
                    epochs=num_epochs, batch_size=512)

history_dict = history.history
print(history_dict)
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
epochs = range(1, len(loss_values) + 1)
plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.clf()
acc = history_dict['acc']
val_acc = history_dict['val_acc']
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

results = model.evaluate(test_data, test_target)
print(results)
