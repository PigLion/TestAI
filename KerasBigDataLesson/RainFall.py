import pandas as pd
import numpy as np
from keras import models
from keras import layers
import matplotlib.pyplot as plt
from keras import optimizers
from keras import losses
from keras import metrics

rainData = pd.read_csv('../../_data/weatherAUS.csv')
rainData= rainData.drop(columns=['Cloud3pm','Cloud9am','Location','RISK_MM','Date'],axis=1)
rainData= rainData.dropna(how='any')
rainData['RainToday'].replace({'No': 0, 'Yes':1},inplace = True)
rainData['RainTomorrow'].replace({'No': 0, 'Yes': 1},inplace = True)
rainData['WindGustDir'].replace({'E':0.06125,'ENE':0.1225,'ESE':0.18375,'N':0.245,'NE':0.30625,'NNE':0.3675,'NNW':0.42875,
                                 'NW':0.49,'S':0.55125,'SE':0.6125,'SSE':0.67375,'SSW':0.735,'SW':0.79625,'W':0.8575,'WNW':0.91875,
                                 'WSW':0.98},inplace = True)
rainData['WindDir3pm'].replace({'E':0.06125,'ENE':0.1225,'ESE':0.18375,'N':0.245,'NE':0.30625,'NNE':0.3675,'NNW':0.42875,
                                'NW':0.49,'S':0.55125,'SE':0.6125,'SSE':0.67375,'SSW':0.735,'SW':0.79625,'W':0.8575,'WNW':0.91875,
                                'WSW':0.98},inplace = True)
rainData['WindDir9am'].replace({'E':0.06125,'ENE':0.1225,'ESE':0.18375,'N':0.245,'NE':0.30625,'NNE':0.3675,'NNW':0.42875,
                                'NW':0.49,'S':0.55125,'SE':0.6125,'SSE':0.67375,'SSW':0.735,'SW':0.79625,'W':0.8575,'WNW':0.91875,
                                'WSW':0.98},inplace = True)
rainToday=rainData['RainToday']
rainTomorrow=rainData['RainTomorrow']
rainData= rainData.drop(columns=['RainTomorrow','RainToday'],axis=1)
target=rainTomorrow
mean = rainData.mean(axis=0)
rainData -= mean
std = rainData.std(axis=0)
rainData /= std
rainData['RainToday']=rainToday
print(rainData[:20])



model = models.Sequential()
model.add(layers.Dense(38, activation='softmax',
                       input_shape=(rainData.shape[1],)))
model.add(layers.Dense(38, activation='softmax'))
model.add(layers.Dense(1,activation='sigmoid'))
model.compile(optimizer=optimizers.RMSprop(lr=0.001),
              loss=losses.binary_crossentropy,
              metrics=[metrics.binary_accuracy])

print(rainData.shape)
print(target.shape)

train_data=rainData[:45000]
test_data=rainData[45000:55000]
valiation_data=rainData[55000:]
train_target=target[:45000]
test_target=target[45000:55000]
validation_target=target[55000:]

printDataAmount=10
printData=rainData[55000:55000+printDataAmount]
printTarget=target[55000:55000+printDataAmount]




num_epochs = 10

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
acc = history_dict['binary_accuracy']
val_acc = history_dict['val_binary_accuracy']
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

results = model.evaluate(test_data, test_target)
print(results)

print(model.predict(printData))
print(printTarget)