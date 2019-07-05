import numpy as np
from pprint import pprint
class Perceptron:
    def __init__(self,data):
        self.data=data
        self.weights=np.zeros(len(data[0]))
        self.bias=0
        self.rate=0.1
    def predict(self,data):
        summation=np.dot(data,self.weights)+self.bias
        if summation < 0:
            summation= -1
        else:
            summation = 1
        return summation
    def train(self,data,labels):
        diff=np.ones(len(data))
        epoch=0
        while diff.any()!=0:
        # for _ in range(10):
            prediction=[]
            for inputs,label in zip(data,labels):
                pred=self.predict(inputs)
                self.bias+=self.rate*(label-pred)
                self.weights+=self.rate*(label-pred)*inputs
                prediction.append(pred)
            diff=(labels-prediction)
            mse=np.sum(diff**2)
            print(mse)
            print("epoch: "+ str(epoch) +"\tdifference:"+str(mse) +"\tweights:"+str(sum(self.weights))+"\tBias:"+str(self.bias))
            epoch += 1

train_data=np.array([
    [0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0],
    [0,1,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,1,1,1,0],
    [0,0,1,0,0,0,1,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0],
    [0,0,0,1,0,0,0,1,1,0,0,0,1,0,0,0,1,1,0,0,0,1,0,0,0],
    [0,1,1,1,0,0,1,0,1,0,0,1,0,1,0,0,1,0,1,0,0,1,1,1,0],
    [0,0,1,0,0,0,1,0,1,0,0,1,0,1,0,0,1,0,1,0,0,0,1,0,0],
    [0,0,1,0,0,0,1,0,1,0,0,1,0,1,0,0,1,0,1,0,0,1,1,1,0],
    [0,0,1,1,0,0,1,0,0,1,0,1,0,0,1,0,1,0,1,0,0,1,1,0,0]
])
train_labels=np.array([1,1,1,1,-1,-1,-1,-1])
test_data=np.array([
    [0,0,0,0,1,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0],
    [0,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,0],
    [0,1,1,1,0,1,0,0,0,1,1,0,0,0,1,1,0,0,0,1,0,1,1,1,0],
    [1,1,1,1,1,1,0,0,0,1,1,0,1,0,1,1,0,0,0,1,1,1,1,1,1]
])
test_labels=np.array([1,1,-1,-1])
p=Perceptron(train_data)
p.train(train_data,train_labels)

predict_labels=[]
for x in test_data:
    predict_labels.append(p.predict(x))
pprint(predict_labels)
