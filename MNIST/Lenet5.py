# from comet_ml import Experiment
# experiment = Experiment()
"""
INPUT => CONV => RELU => POOL => CONV => RELU => POOL => FC => RELU => FC
"""
import matplotlib.pyplot as plt
import time
from keras import models, layers , backend as K
# from keras import backend as K

from keras.datasets import mnist
from keras.utils import np_utils
import numpy as np
import winsound
# import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# NUM_PARALLEL_EXEC_UNITS = 4
# config = tf.ConfigProto(intra_op_parallelism_threads=NUM_PARALLEL_EXEC_UNITS, inter_op_parallelism_threads=2,
#                        allow_soft_placement=True, device_count={'CPU': NUM_PARALLEL_EXEC_UNITS})
# session = tf.Session(config=config)
# K.set_session(session)

def plot_digit(X, y, idx):
    img = X[idx].reshape(28,28)
    plt.imshow(img, cmap='Greys',  interpolation='nearest')
    plt.annotate(y[idx], xy=(0.05, 0.85), xycoords="axes fraction", color="blue", fontsize=23)
    plt.show()

tic=time.time()
(train_images,train_labels), (test_images,test_labels) = mnist.load_data()

train_images = np.array(train_images)
test_images = np.array(test_images)

#Reshape the training and test set
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)
test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)

# train_images = np.pad(train_images, ((0,0),(2,2),(2,2),(0,0)), 'constant')
# test_images = np.pad(test_images, ((0,0),(2,2),(2,2),(0,0)), 'constant')

# Normalize
train_images = train_images.astype('float32')/255
test_images = test_images.astype('float32')/255

train_labels = np_utils.to_categorical(train_labels, 10)
test_labels = np_utils.to_categorical(test_labels, 10)

print(np.array(train_labels).shape)

"""
Convolution #1. Input = 32x32x1. Output = 28x28x6 conv2d
SubSampling #1. Input = 28x28x6. Output = 14x14x6. SubSampling is simply Average Pooling so we use avg_pool
Convolution #2. Input = 14x14x6. Output = 10x10x16 conv2d
SubSampling #2. Input = 10x10x16. Output = 5x5x16 avg_pool
Fully Connected #1. Input = 5x5x16. Output = 120
Fully Connected #2. Input = 120. Output = 84
Output 10
"""


#
model = models.Sequential()

model.add(layers.Conv2D(filters = 6,kernel_size=(5, 5), strides=1,activation = 'relu',input_shape = (28,28,1)))

model.add(layers.AveragePooling2D(pool_size = 2, strides = 2, padding='valid'))

model.add(layers.Conv2D(filters = 16,kernel_size=(5, 5), strides=1,activation = 'relu',input_shape = (14,14,6)))

model.add(layers.AveragePooling2D(pool_size = 2, strides = 2))

model.add(layers.Flatten())

model.add(layers.Dense(units = 120, activation = 'relu'))

model.add(layers.Dense(units = 84, activation = 'relu'))

model.add(layers.Dense(units = 10, activation = 'softmax'))

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
#
hist = model.fit(x=train_images,y=train_labels,
                 epochs=10, batch_size=128,
                 validation_data=(test_images, test_labels),
                 verbose=10)
#
test_score = model.evaluate(test_images, test_labels)
print("Test loss {:.4f}, accuracy {:.2f}%".format(test_score[0], test_score[1] * 100))
#
#
f, ax = plt.subplots()
ax.plot([None] + hist.history['acc'], 'o-')
ax.plot([None] + hist.history['val_acc'], 'x-')
ax.legend(['Train acc', 'Validation acc'], loc = 'best')
ax.set_title('Training/Validation acc per Epoch')
ax.set_xlabel('Epoch')
ax.set_ylabel('acc')
plt.show()

f, ax = plt.subplots()
ax.plot([None] + hist.history['loss'], 'o-')
ax.plot([None] + hist.history['val_loss'], 'x-')
ax.legend(['Train Loss', 'Validation Loss'], loc = 'best')
ax.set_title('Training/Validation Loss per Epoch')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')

plt.show()
print(str(time.time() - tic) + ' s')
winsound.Beep(500,1000)
