import keras
import matplotlib.pyplot as plt
import numpy as np
(train_images,train_labels),(test_images,test_labels)=keras.datasets.mnist.load_data()

base_path=r"C:\Users\Muge\OneDrive\Documents\School\MSc. CaSP\Class Reference\EEEN60070 Dissertation\Core\images"
#%%
cnt=100
cols,x,y=test_images.shape
choices=np.random.choice(a=cols,size=cnt)

fig = plt.figure(figsize=(15,15))
fig.subplots_adjust(hspace=0, wspace=0)
for i in range(cnt):
    # print(i)
    ax = fig.add_subplot(10, 10, i+1)
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(ax.get_yticklabels(), visible=False)
    plt.axis('off')
    # ax.text(0.5, 0.5, str((2, 3, i)),fontsize=18, ha='center')
    ax.imshow(np.asarray(test_images[choices[i]]),cmap = 'gray')
plt.show()
# plt.draw()
plt.savefig(base_path+'\mnist_dataset.png', format='png',bbox_inches='tight')

#%%
import EigenFaces.EigenFaces as eig
import importlib
importlib.reload(eig)
train_data,train_label,test_data,test_label,full_data,full_label=eig.load_images()
print(train_data.shape)
#%%
cnt=25
# print(np.array(train_data[0]).reshape(46,56).shape)
choices=np.random.choice(a=train_data.shape[0],size=cnt)
fig = plt.figure(figsize=(30,30))
fig.subplots_adjust(hspace=0, wspace=0)
for i in range(cnt):
    # print(i)
    ax = fig.add_subplot(5, 5, i+1)
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(ax.get_yticklabels(), visible=False)
    plt.axis('off')
    ax.imshow(np.array(train_data[choices[i]]).reshape(46,56),cmap = 'gray')
plt.show()