from comet_ml import Experiment
# Add the following code anywhere in your machine learning file
# experiment = Experiment()

import winsound
import keras
import time
import MLP

tic=time.time()
(train_images,train_labels),(test_images,test_labels)=keras.datasets.mnist.load_data()
train_images=train_images.reshape((train_images.shape[0],train_images.shape[1]*train_images.shape[1])).astype(float)
test_images=test_images.reshape((test_images.shape[0],test_images.shape[1]*test_images.shape[1])).astype(float)
train_images /=255
test_images /=255

# dataset=np.array(list(zip(test_images[0:2000],test_labels[0:2000])))
# test_data=np.array(list(zip(test_images[2000:2500],test_labels[2000:2500])))

print(str(time.time() - tic) + ' s')
winsound.Beep(500,1000)

