import pandas as pd
import numpy as np
from keras import models
from keras import layers
import matplotlib.pyplot as plt
from keras import optimizers
from keras import losses
from keras import metrics

heartData = pd.read_csv('../../_data/heart.csv')
heartData= heartData.dropna(how='any')
target=heartData.loc[:,'target']
heartData=heartData.drop(columns=['target'],axis=1)
mean = heartData.mean(axis=0)
heartData -= mean
std = heartData.std(axis=0)
heartData /= std
print(heartData[:20])
def build_model():
    model = models.Sequential()
    model.add(layers.Dense(38, activation='sigmoid',
                           input_shape=(heartData.shape[1],)))
    model.add(layers.Dense(38, activation='sigmoid'))
    model.add(layers.Dense(1,activation='sigmoid'))
    model.compile(optimizer=optimizers.RMSprop(lr=0.0005),
                  loss=losses.binary_crossentropy,
                  metrics=[metrics.binary_accuracy])
    return model


train_data=heartData[50:]
test_data=heartData[:50]
train_target=target[50:]
test_target=target[:50]



num_epochs = 15
k = 4
num_val_samples = len(train_data) // k
all_loss_histories = []
all_Valloss_histories = []
all_acc_histories = []
all_Valacc_histories = []
model=build_model()
for i in range(k):
    print('processing fold #', i)
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_target[i * num_val_samples: (i + 1) * num_val_samples]
    partial_train_data = np.concatenate(
        [train_data[:i * num_val_samples],
         train_data[(i + 1) * num_val_samples:]],
        axis=0)
    partial_target = np.concatenate(
        [train_target[:i * num_val_samples],
         train_target[(i + 1) * num_val_samples:]],
        axis=0)
    history = model.fit(partial_train_data, partial_target,
                        validation_data=(val_data, val_targets),
                        epochs=num_epochs, batch_size=1)
    print(history.history)
    all_loss_histories.append(history.history['loss'])
    all_Valloss_histories.append(history.history['val_loss'])
    all_acc_histories.append(history.history['binary_accuracy'])
    all_Valacc_histories.append(history.history['val_binary_accuracy'])

all_loss_histories = [np.mean([x[i] for x in all_loss_histories]) for i in range(num_epochs)]
all_Valloss_histories = [np.mean([x[i] for x in  all_Valloss_histories]) for i in range(num_epochs)]
all_acc_histories = [np.mean([x[i] for x in all_acc_histories]) for i in range(num_epochs)]
all_Valacc_histories = [np.mean([x[i] for x in all_Valacc_histories]) for i in range(num_epochs)]


loss_values = all_loss_histories
val_loss_values =  all_Valloss_histories
epochs = range(1, len(loss_values) + 1)
plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.clf()
acc = all_acc_histories
val_acc = all_Valacc_histories
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

results = model.evaluate(test_data, test_target)
print(results)
