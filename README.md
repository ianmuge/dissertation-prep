#Preparation
##Perceptron
###Single neuron
Performance after convergence, in this case 2 epochs, does not vary. No changes after convergence. Final weight sum is 0. Final Bias is dependent on the learning rate, 0.2
###SLP
Over 300 Epochs with a learning rate of 0.25, we manage to accomplish a test Accuracy: 75% in 0.6996002197265625 s. Input has the same nodes as the number of cross-cut attributes being analysed. We have three neurons in the hidden layer and  the output has a set of unique labels
##EigenFaces
We attain an accuracy of about 90.5%
##MNIST
###Lenet5
- Convolution #1. Input = 32x32x1. Output = 28x28x6 conv2d
- SubSampling #1. Input = 28x28x6. Output = 14x14x6. SubSampling is simply Average Pooling so we use avg_pool
- Convolution #2. Input = 14x14x6. Output = 10x10x16 conv2d
- SubSampling #2. Input = 10x10x16. Output = 5x5x16 avg_pool
- Fully Connected #1. Input = 5x5x16. Output = 120
- Fully Connected #2. Input = 120. Output = 84
- Output 10

suggested activation function is tanh, ReLU observed to have a higher accuracy  
Test loss 0.0381, accuracy 98.72% \
Completed:244.60259580612183 s

###SKLearn Classifier
Training set score: 0.996483 \
Test set score: 0.966400\
    precision    recall  f1-score   support\
           0       0.98      0.99      0.98       980\
           1       0.99      0.99      0.99      1135\
           2       0.96      0.97      0.96      1032\
           3       0.95      0.96      0.95      1010\
           4       0.97      0.97      0.97       982\
           5       0.97      0.95      0.96       892\
           6       0.97      0.97      0.97       958\
           7       0.97      0.96      0.96      1028\
           8       0.96      0.95      0.95       974\
           9       0.96      0.96      0.96      1009
###Vanilla MLP
Hidden Layers 3 \
Epochs 25\
Train accuracy: 0.99\
Val accuracy: 0.9335\
Execution time: 26.512019634246826 s
##ViolaJones
##YOLO
To get the full yolo weights 
curl -o "./Yolo/data/yolo.weights" -XGET https://pjreddie.com/media/files/yolov3.weights

