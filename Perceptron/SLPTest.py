import time
from Perceptron import SLP

dataset=[
    [0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,1],
    [0,1,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,1,1,1,0,1],
    [0,0,1,0,0,0,1,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,1],
    [0,0,0,1,0,0,0,1,1,0,0,0,1,0,0,0,1,1,0,0,0,1,0,0,0,1],
    [0,1,1,1,0,0,1,0,1,0,0,1,0,1,0,0,1,0,1,0,0,1,1,1,0,0],
    [0,0,1,0,0,0,1,0,1,0,0,1,0,1,0,0,1,0,1,0,0,0,1,0,0,0],
    [0,0,1,0,0,0,1,0,1,0,0,1,0,1,0,0,1,0,1,0,0,1,1,1,0,0],
    [0,0,1,1,0,0,1,0,0,1,0,1,0,0,1,0,1,0,1,0,0,1,1,0,0,0]
]

test_data=[
    [0,0,0,0,1,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,1],
    [0,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,0,1],
    [0,1,1,1,0,1,0,0,0,1,1,0,0,0,1,1,0,0,0,1,0,1,1,1,0,0],
    [1,1,1,1,1,1,0,0,0,1,1,0,1,0,1,1,0,0,0,1,1,1,1,1,1,0]
]


tic=time.time()
n_inputs = len(dataset[0]) - 1
n_outputs = len(set([row[-1] for row in dataset]))

network = SLP.SLP(n_inputs, 3, n_outputs)
network.train_network(network.network, dataset, 0.5, 100, n_outputs)
# for layer in network.network:
#     print(layer)
data_acc_count=0
for row in dataset:
    prediction = network.predict(network.network, row)
    print('Expected=%d, Got=%d' % (row[-1], prediction))
    if int(row[-1])== int(prediction):
        data_acc_count+=1

print("Accuracy: %d"% (float(data_acc_count/len(dataset))*100))
test_data_acc_count=0
for row in test_data:
    prediction = network.predict(network.network, row)
    print('Expected=%d, Got=%d' % (row[-1], prediction))
    if int(row[-1])== int(prediction):
        test_data_acc_count+=1
print("Test Accuracy: %d"% (float(test_data_acc_count/len(test_data))*100))
print(str(time.time() - tic) + ' s')

