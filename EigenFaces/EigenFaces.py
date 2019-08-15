"""
Eigenfaces training and analysis code
"""
#Setup
import numpy as np
from skimage.transform import resize
from skimage import io
import matplotlib.pyplot as plt
import time
import warnings
warnings.filterwarnings("ignore")

img_repo="/mnt/c/Users/Muge/PycharmProjects/DissertationWorkspace/Preparation/EigenFaces/data/faces/"

def load_images():
    train_data=[]
    train_label=[]
    test_data=[]
    test_label=[]
    full_data=[]
    full_label=[]
    train_num=5
    for i in np.linspace(1, 40, 40):
        for j in np.linspace(1, 10, 10):
            filename = img_repo + '/s' + str(i.astype(np.int8)) + '/' + str(j.astype(np.int8)) + '.pgm'
            imvector=img_vector(filename)
            if j <= train_num:
                train_data.append(imvector)
                train_label.append(i)
            else:
                test_data.append(imvector)
                test_label.append(i)
            full_data.append(imvector)
            full_label.append(i)
    return np.array(train_data),np.array(train_label), \
           np.array(test_data),np.array(test_label), \
           np.array(full_data),np.array(full_label)
# def plot_imgs():
#     fig, axs = plt.subplots(40,10,figsize=(20,80))
#     fig.subplots_adjust(hspace=0.4, wspace=0.4)
#     for i in range(40):
#         for j in range(10):
#             filename = img_repo + 's' + str(i+1) + '/' + str(j+1) + '.pgm'
#             axs[i,j].imshow(plt.imread(filename), cmap='gray')
#             axs[i,j].axis('off')
#     plt.tight_layout()
#     plt.axis('off')
#     plt.show()

def img_vector(img):
    return resize(io.imread(img),(46,56)).flatten()


def subvector(target_matrix, target_vector):
    vector4matrix = np.repeat(target_vector, target_matrix.shape[0], axis=0)
    target_matrix = target_matrix - vector4matrix
    return target_matrix

def submean(data):
    data=np.array(data)
    mean_data = data.mean(axis=0).reshape(1, data.shape[1])
    return subvector(data, mean_data)

def pca(data,limit=0):
    [n, d] = data.shape
    if (limit <= 0) or (limit > n):
        limit = n
    mu = data.mean(axis=0)
    data = data - mu
    cov = np.dot(data.T, data)
    l, v = np.linalg.eig(cov)
    idx = np.argsort(- l)
    l = l[idx]
    v = v[:, idx]
    l = l[0: limit].copy()
    v = v[:, 0: limit].copy()
    return l,v,mu
def project (W , X , mu = None ):
    if mu is None :
        return np.dot (X ,W)
    return np . dot (X - mu , W)
def reconstruct (W , Y , mu = None ) :

    if mu is None :
        return np . dot (Y ,W.T)
    return np . dot (Y , W .T) + mu
def normalize (X , low , high):
    X = np . asarray (X)
    minX , maxX = np . min (X ) , np . max (X)

    X = X - float ( minX )
    X = X / float (( maxX - minX ) )

    X = X * ( high - low )
    X = X + low
    return np.asarray(X,dtype='float')

def subplot ( title , images , rows , cols , sptitle =" subplot " ,sptitles =[] ):
    fig = plt.figure(figsize=(5,5))
    # fig.text(.5, .95, title, horizontalalignment='center')
    fig.suptitle(title, fontsize=10)
    fig.subplots_adjust(hspace=0, wspace=0)
    for i in range(len(images)):
        ax0 = fig.add_subplot ( rows , cols ,(i+1 ))
        plt.setp(ax0.get_xticklabels () , visible = False )
        plt.setp(ax0.get_yticklabels () , visible = False )
        plt.axis('off')
        if len(sptitles) == len(images):
            plt.title("%s #%s" % (sptitle, str(sptitles[i])),fontsize=5)
        else:
            plt.title("%s #%d" % (sptitle, (i + 1)),fontsize=5)
        plt.imshow(np.asarray(images[i]),cmap = 'gray')
    plt.show ()

#%%
#Training

train_data,train_label,test_data,test_label,full_data,full_label=load_images()
tic=time.time()
train_l,train_v,mean=pca(train_data,1000)
print("Training time:",time.time()-tic)
#%%
#Plots

fig=plt.figure()
plt.title("Mean Face")
plt.axis('off')
plt.imshow(np.asarray(normalize(mean.reshape((46,56)),0,255)),cmap = 'gray')
plt.show()
E = []
for i in range (min(len(train_data),20)):
    e = train_v[:,i].reshape((46,56))
    E.append(normalize(e,0,255))
subplot(title ="", images =E , rows =5 , cols =4 , sptitle ="")

#%%
# steps =[ i for i in range (10 , min ( len (train_data) , 320) , 20)]
steps =[ i for i in range (0 , 1000 , 50)]
E = []
for i in range(min(len(steps),20)):
    numEvs = steps [i]
    P = project (train_v [: ,0: numEvs ], train_data [0]. reshape (1 ,-1) , mean )
    R = reconstruct (train_v [: ,0: numEvs ], P , mean )
    R = R . reshape ( (46,56))
    E. append ( normalize (R ,0 ,255) )

subplot ( title ="", images =E , rows =4 , cols =5 , sptitle ="", sptitles = steps)

#%%
train_v = train_v[:,0:int(train_v.shape[1])]
train_data = np.dot(train_data, train_v)
test_data = np.dot(test_data , train_v)
#
#
tic=time.time()
count = 0
for i in np.linspace(0, test_data.shape[0] - 1, test_data.shape[0]).astype(np.int64):
    sub = subvector(train_data,test_data[i, :].reshape((1,test_data.shape[1])))
    dis = np.linalg.norm(sub, axis = 1)
    fig = np.argmin(dis)
    if train_label[fig] == test_label[i]:
        count = count + 1
print("Validation Time:",time.time()-tic)
correct_rate = count / test_data.shape[0]

print("Accuracy rate:", correct_rate * 100 , "%")
#%%


fig = plt.figure(figsize=(15,15))
# fig.text(.5, .95, title, horizontalalignment='center')
fig.suptitle("Feature Maps", fontsize=30)
fig.subplots_adjust(hspace=0, wspace=0)

r, c = (5, 8)
for i in range(r * c):
    plt.subplot(r,c,i+1)
    plt.imshow(train_v[:, i].real.reshape(46, 56), cmap='gray')
    plt.axis('off')
plt.show()
#%%
"""
Independent Analysis
"""
tic=time.time()
train_l,train_v,mean=pca(train_data,200)
print(time.time()-tic)
tic=time.time()
train_l_,train_v_,mean_=pca(train_data,600)
print(time.time()-tic)
