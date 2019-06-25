import numpy as np
import pandas as pd
import pprint
from skimage.transform import resize
from skimage import io
# import matplotlib.image as mpimg

class Eigenfaces:
    def __init__(self,img_repo):
        self.img_repo=img_repo
    def load_images(self):
        train_data=[]
        train_label=[]
        test_data=[]
        test_label=[]
        train_num=5
        for i in np.linspace(1, 40, 40):  # 40 sample people
            for j in np.linspace(1, 10, 10):  # everyone has 10 different face
                filename = self.img_repo + '/s' + str(i.astype(np.int8)) + '/' + str(j.astype(np.int8)) + '.pgm'
                imvector=self.img_vector(filename)
                if j <= train_num:
                    train_data.append(imvector)
                    train_label.append(i)
                else:
                    test_data.append(imvector)
                    test_label.append(i)
        return train_data,train_label,test_data,test_label
    def img_vector(self,img):
        img=resize(io.imread(img),(2576,1)).flatten()
        return img

    def subvector(self,target_matrix, target_vector):
        vector4matrix = np.repeat(target_vector, target_matrix.shape[0], axis=0)
        target_matrix = target_matrix - vector4matrix
        return target_matrix
    def submean(self,data,test):
        data=np.array(data)
        test=np.array(test)
        mean_data = data.mean(axis=0).reshape(1, data.shape[1])
        train_data = self.subvector(data, mean_data)
        test_data = self.subvector(test, mean_data)
        return train_data,test_data
    def cov_mat(self,data):
        cov = np.dot(data.T, data)
        l, v = np.linalg.eig(cov)
        return l,v
eg=Eigenfaces(img_repo="./data/faces/")
train_data,train_label,test_data,test_label=eg.load_images()
train_data,test_data=eg.submean(train_data,test_data)

train_l,train_v=eg.cov_mat(train_data)

mix = np.vstack((train_l,train_v))
mix = mix.T[np.lexsort(mix[::-1,:])].T[:,::-1]
train_v = np.delete(mix, 0, axis=0)

train_v = train_v[:,0:int(train_v.shape[1])]
train_data = np.dot(train_data, train_v)
test_data = np.dot(test_data , train_v)
