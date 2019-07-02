from comet_ml import Experiment
experiment = Experiment()
"""
INPUT => CONV => RELU => POOL => CONV => RELU => POOL => FC => RELU => FC
"""
import matplotlib.pyplot as plt
import time
from keras import models, layers
from keras.datasets import mnist
from keras.utils import np_utils
import numpy as np

from keras import backend as K
import tensorflow as tf
NUM_PARALLEL_EXEC_UNITS = 4
config = tf.ConfigProto(intra_op_parallelism_threads=NUM_PARALLEL_EXEC_UNITS, inter_op_parallelism_threads=2,
                       allow_soft_placement=True, device_count={'CPU': NUM_PARALLEL_EXEC_UNITS})
session = tf.Session(config=config)
K.set_session(session)


def plot_digit(X, y, idx):
    img = X[idx].reshape(28,28)
    plt.imshow(img, cmap='Greys',  interpolation='nearest')
    plt.annotate(y[idx], xy=(0.05, 0.85), xycoords="axes fraction", color="blue", fontsize=23)
    # plt.title('true label: %d' % y[idx])
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

train_images = train_images.astype('float32')
test_images = test_images.astype('float32')

train_labels = np_utils.to_categorical(train_labels, 10)
test_labels = np_utils.to_categorical(test_labels, 10)

#
# # Normalize value to [0, 1]
train_images /= 255
test_images /= 255
#
#
model = models.Sequential()
#
# #Layer 1
# #Conv Layer 1
model.add(layers.Conv2D(filters = 6,kernel_size=(5, 5), strides=1,activation = 'relu',input_shape = (28,28,1)))
# #Pooling layer 1
model.add(layers.AveragePooling2D(pool_size = 2, strides = 2, padding='valid'))
# #Layer 2
# #Conv Layer 2
model.add(layers.Conv2D(filters = 16,kernel_size=(5, 5), strides=1,activation = 'relu',input_shape = (14,14,6)))
# #Pooling Layer 2
model.add(layers.AveragePooling2D(pool_size = 2, strides = 2))
#
# # # C5 Fully Connected Convolutional Layer
# model.add(layers.Conv2D(120, kernel_size=(5, 5), strides=(1, 1), activation='tanh', padding='valid'))
#
# #Flatten
model.add(layers.Flatten())
# #Layer 3
# #Fully connected layer 1
model.add(layers.Dense(units = 120, activation = 'relu'))
# #Layer 4
# #Fully connected layer 2
model.add(layers.Dense(units = 84, activation = 'relu'))
# #Layer 5
# #Output Layer
model.add(layers.Dense(units = 10, activation = 'softmax'))
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
#
# hist=model.fit(train_images ,train_labels, steps_per_epoch = 10, epochs = 42)
hist = model.fit(x=train_images,y=train_labels, epochs=10, batch_size=128, validation_data=(test_images, test_labels), verbose=10)
#
test_score = model.evaluate(test_images, test_labels)
print("Test loss {:.4f}, accuracy {:.2f}%".format(test_score[0], test_score[1] * 100))
#
#
f, ax = plt.subplots()
ax.plot([None] + hist.history['acc'], 'o-')
ax.plot([None] + hist.history['val_acc'], 'x-')
# Plot legend and use the best location automatically: loc = 0.
ax.legend(['Train acc', 'Validation acc'], loc = 0)
ax.set_title('Training/Validation acc per Epoch')
ax.set_xlabel('Epoch')
ax.set_ylabel('acc')

f, ax = plt.subplots()
ax.plot([None] + hist.history['loss'], 'o-')
ax.plot([None] + hist.history['val_loss'], 'x-')
# Plot legend and use the best location automatically: loc = 0.
ax.legend(['Train Loss', 'Validation Loss'], loc = 0)
ax.set_title('Training/Validation Loss per Epoch')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')

plt.show()
print('Completed:'+str(time.time() - tic) + ' s')
