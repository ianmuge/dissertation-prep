import keras
import matplotlib.pyplot as plt
import time
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
import numpy as np

def plot_digit(X, y, idx):
    img = X[idx].reshape(28,28)
    plt.imshow(img, cmap='Greys',  interpolation='nearest')
    plt.annotate(y[idx], xy=(0.05, 0.85), xycoords="axes fraction", color="blue", fontsize=23)
    plt.show()


(train_images,train_labels),(test_images,test_labels)=keras.datasets.mnist.load_data()
train_images=train_images.reshape((train_images.shape[0],train_images.shape[1]*train_images.shape[1])).astype(float)
test_images=test_images.reshape((test_images.shape[0],test_images.shape[1]*test_images.shape[1])).astype(float)
train_images /=255
test_images /=255

mlp = MLPClassifier(hidden_layer_sizes=(25,50,25), #(100,),
                    activation='relu', #relu|tanh|logistic|identity
                    max_iter=25, alpha=1e-4,
                    batch_size='auto',
                    solver='adam', verbose=10, #ibfgs|sgd|adam
                    tol=1e-4, random_state=1,
                    warm_start=True,
                    early_stopping=True,
                    validation_fraction=0.2)
# learning_rate='adaptive',learning_rate_init=0.5) #constant|invscaling|adaptive
tic=time.time()
mlp.fit(train_images, train_labels)
print('Training:'+str(time.time() - tic) + ' s')
print("Training set score: %f" % mlp.score(train_images, train_labels))
print("Test set score: %f" % mlp.score(test_images, test_labels))
#%%

#%%
cnt=100
choices=np.random.choice(a=test_images.shape[0],size=cnt)
tic=time.time()
test_pred=mlp.predict(test_images[choices])
print('Prediction:'+str(time.time() - tic) + ' s')
print(classification_report(test_labels[choices],test_pred ))
print(accuracy_score(test_labels[choices], test_pred))
print(confusion_matrix(test_labels[choices], test_pred))

# plt.figure()
# plt.imshow(confusion_matrix(test_labels, test_pred), cmap='hot', interpolation='nearest')
# plt.show()
#%%
plt.title("SKLearn MLP loss")
plt.plot(mlp.loss_curve_, label='loss')
plt.legend(loc='best')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.grid()
plt.show()
print('Completed:'+str(time.time() - tic) + ' s')

