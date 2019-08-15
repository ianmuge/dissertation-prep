import time
from MNIST.MLP import *
import numpy as np
import matplotlib.pyplot as plt
import keras


mlp=MLP()
(train_data, train_label), (test_data, test_label) = keras.datasets.mnist.load_data()
# x_train,y_train,x_test,y_test = mlp.load_dataset(train_data, train_label,test_data, test_label,(50000,10000))
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
# network.append(Dense(200,300))
# network.append(ReLU())
# network.append(Dense(300,200))
# network.append(ReLU())
network.append(Dense(200,len(set(y_train))))

tic=time.time()
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
print("Train time:"+str(time.time() - tic) + ' s')

#%%
plt.title("Baseline MLP training and Validation Accuracy")
plt.plot(train_log, label='Training accuracy')
plt.plot(val_log, label='Validation accuracy')
plt.legend(loc='best')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.grid()
plt.show()
"""
Epoch 24
Train accuracy: 0.9998333333333334
Val accuracy: 0.9235
12.383655786514282 s
"""
#%%
# cnt=100
# print(np.array(train_data[0]).reshape(46,56).shape)
# choices=np.random.choice(a=x_test.shape[0],size=cnt)

tic=time.time()
print(np.mean(mlp.predict(network, x_test) == y_test))
print("Test time:"+str(time.time() - tic) + ' s')

