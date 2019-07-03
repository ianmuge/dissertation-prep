from comet_ml import Experiment
# Add the following code anywhere in your machine learning file
# experiment = Experiment()

import winsound
import keras
import time
from MLP import *
import numpy as np
import matplotlib.pyplot as plt
import keras
from tqdm import trange

tic=time.time()
mlp=MLP()
(train_data, train_label), (test_data, test_label) = keras.datasets.mnist.load_data()
x_train,y_train,x_test,y_test = mlp.load_dataset(train_data, train_label,test_data, test_label)

# plt.figure(figsize=[6,6])
# for i in range(4):
#     plt.subplot(2,2,i+1)
#     plt.title("Label: %i"%y_train[i])
#     plt.imshow(x_train[i].reshape([28,28]),cmap='gray')
# plt.show()

network = []
network.append(Dense(x_train.shape[1],100))
network.append(ReLU())
network.append(Dense(100,200))
network.append(ReLU())
network.append(Dense(200,len(set(y_train))))


train_log = []
val_log = []
for epoch in range(25):

    for x_batch, y_batch in mlp.iterate_minibatches(x_train, y_train, batchsize=50):
        mlp.train(network, x_batch, y_batch)

    train_log.append(np.mean(mlp.predict(network, x_train) == y_train))
    val_log.append(np.mean(mlp.predict(network, x_test) == y_test))

    print("Epoch", epoch)
    print("Train accuracy:", train_log[-1])
    print("Val accuracy:", val_log[-1])

plt.plot(train_log, label='train accuracy')
plt.plot(val_log, label='val accuracy')
plt.legend(loc='best')
plt.grid()
plt.show()

print(str(time.time() - tic) + ' s')
winsound.Beep(500,1000)
