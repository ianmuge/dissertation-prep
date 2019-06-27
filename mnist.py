from comet_ml import Experiment
# Add the following code anywhere in your machine learning file
experiment = Experiment(api_key="rqDYIkwKDvdj9pKIGT1MWZTpX",
                        project_name="dissertation-prep",
                        workspace="imuge")

import pandas as pd
import numpy as np
from mlxtend.data import loadlocal_mnist
import matplotlib.pyplot as plt
import time
from sklearn.neural_network import MLPClassifier
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

def plot_digit(X, y, idx):
    img = X[idx].reshape(28,28)
    plt.imshow(img, cmap='Greys',  interpolation='nearest')
    plt.annotate(y[idx], xy=(0.05, 0.85), xycoords="axes fraction", color="red", fontsize=23)
    # plt.title('true label: %d' % y[idx])
    plt.show()

tic=time.time()
train_images,train_labels = loadlocal_mnist(
    images_path='./data/train-images.idx3-ubyte',
    labels_path='./data/train-labels.idx1-ubyte')

test_images,test_labels  = loadlocal_mnist(
    images_path='./data/t10k-images.idx3-ubyte',
    labels_path='./data/t10k-labels.idx1-ubyte')

# plot_digit(train_images,train_labels,2)

mlp = MLPClassifier(hidden_layer_sizes=(25,50,25), #(100,),
                    activation='relu', #relu|tanh|logistic|identity
                    max_iter=500, alpha=1e-4,
                    batch_size='auto',
                    solver='sgd', verbose=10, #ibfgs|sgd|adam
                    tol=1e-4, random_state=1,
                    warm_start=True)
# learning_rate='adaptive',learning_rate_init=0.5) #constant|invscaling|adaptive

mlp.fit(train_images, train_labels)

# with ThreadPoolExecutor(4) as executor:
#     executor.map(mlp.fit, (train_images, train_labels))
# with Pool(4) as p:
#     mlp.fit(train_images, train_labels)


print("Training set score: %f" % mlp.score(train_images, train_labels))
print("Test set score: %f" % mlp.score(test_images, test_labels))

test_pred=mlp.predict(test_images)

print(classification_report(test_labels,test_pred ))
print(accuracy_score(test_labels, test_pred))
print(confusion_matrix(test_labels, test_pred))

experiment.log_metric("train_accuracy", mlp.score(train_images, train_labels))
experiment.log_metric("test_accuracy", mlp.score(test_images, test_labels))

# plt.figure()
# plt.imshow(confusion_matrix(test_labels, test_pred), cmap='hot', interpolation='nearest')
# plt.show()

plt.figure()
plt.plot(mlp.loss_curve_)
plt.title('Loss Curve')
print('Completed:'+str(time.time() - tic) + ' s')

