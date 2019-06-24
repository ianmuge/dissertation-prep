import numpy as np
import pandas as pd
import os
import pprint

class Eigenfaces:
    def __init__(self,img_repo):
        self.img_repo=img_repo
    def load_images(self):
        files = []
        train_data=[]
        test_data=[]
        for r, d, f in os.walk(self.img_repo):
            for file in f:
                if np.arange(1,6) in file:
                    train_data.append(r+'/'+file)
                else:
                    test_data.append(r + '/' + file)
        return test_data,train_data
    def img_vector(self,img):

        return img
eg=Eigenfaces(img_repo="./data/faces/")
print(eg.load_images()[0])
