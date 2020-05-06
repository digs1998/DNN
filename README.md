# DNN
built a deep network, and apply it to cat vs non-cat classification
Updated now

1 - Packages
Let's first import all the packages that you will need during this assignment.

numpy is the fundamental package for scientific computing with Python.

matplotlib is a library to plot graphs in Python.

h5py is a common package to interact with a dataset that is stored on an H5 file.

PIL and scipy are used here to test your model with your own picture at the end.

dnn_app_utils provides the functions implemented in the "Building your Deep Neural Network: Step by Step" assignment to this notebook.

np.random.seed(1) is used to keep all the random function calls consistent. It will help us grade your work

2.- Datasets
same "Cat vs non-Cat" dataset as in "Logistic Regression as a Neural Network" (Assignment 2). The model you had built had 70% test accuracy on classifying cats vs non-cats images. (will be updating with dataset shortly...)

3 - Architecture of your model
Now that you are familiar with the dataset, it is time to build a deep neural network to distinguish cat images from non-cat images.

You will build two different models:

A 2-layer neural network

An L-layer deep neural network

You will then compare the performance of these models, and also try out different values for L.

A 2-Layer Model will be 
1. The input is a (64,64,3) image which is flattened to a vector of size $(12288,1)$.

2. The corresponding vector: $[x_0,x_1,...,x_{12287}]^T$ is then multiplied by the weight matrix $W^{[1]}$ of size $(n^{[1]}, 12288)$.

3. You then add a bias term and take its relu to get the following vector: $[a_0^{[1]}, a_1^{[1]},..., a_{n^{[1]}-1}^{[1]}]^T$.

You then repeat the same process.


You multiply the resulting vector by $W^{[2]}$ and add your intercept (bias).
Finally, you take the sigmoid of the result. If it is greater than 0.5, you classify it to be a cat.

For an L-Layer it will be difficult to represent exactly what it should contain, but here are some basic points that should be there in the layered model.

-The input is a (64,64,3) image which is flattened to a vector of size (12288,1).

-The corresponding vector: $[x_0,x_1,...,x_{12287}]^T$ is then multiplied by the weight matrix $W^{[1]}$ and then you add the intercept $b^{[1]}$. The result is called the linear unit.

-Next, you take the relu of the linear unit. This process could be repeated several times for each $(W^{[l]}, b^{[l]})$ depending on the model architecture.

-Finally, you take the sigmoid of the final linear unit. If it is greater than 0.5, you classify it to be a cat.

General methodology
As usual you will follow the Deep Learning methodology to build the model:

1. Initialize parameters / Define hyperparameters

2. Loop for num_iterations:
    a. Forward propagation
    b. Compute cost function
    c. Backward propagation
    d. Update parameters (using parameters, and grads from backprop) 
    
3. Use trained parameters to predict labels

Let's now implement those two models!

4 - Two-layer neural network
Question: Use the helper functions you have implemented in the previous assignment to build a 2-layer neural network with the following structure: LINEAR -> RELU -> LINEAR -> SIGMOID. The functions you may need and their inputs are:

def initialize_parameters(n_x, n_h, n_y):
    ...
    return parameters 
def linear_activation_forward(A_prev, W, b, activation):
    ...
    return A, cache
def compute_cost(AL, Y):
    ...
    return cost
def linear_activation_backward(dA, cache, activation):
    ...
    return dA_prev, dW, db
def update_parameters(parameters, grads, learning_rate):
    ...
    return parameters
