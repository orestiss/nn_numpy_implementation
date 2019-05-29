import numpy as np
import matplotlib.pyplot as plt
from .nn_utils import *


class NeuralNet():

    def __init__(self, layers_dims, learning_rate = 0.0075, 
            num_iterations = 300, print_cost=False, output_layer='sigmoid'):
        self.layers_dims = layers_dims
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.print_cost = print_cost
        self.parameters = self.initialize_parameters_deep()
        self.output_layer = output_layer

    def initialize_parameters_deep(self):
        parameters = {}
        L = len(self.layers_dims)  # number of layers in the network

        for l in range(1, L):
            parameters['W' + str(l)] = np.random.randn(self.layers_dims[l], self.layers_dims[l-1]) * 0.01
            parameters['b' + str(l)] = np.zeros((self.layers_dims[l], 1))
            
            assert(parameters['W' + str(l)].shape == (self.layers_dims[l], self.layers_dims[l-1]))
            assert(parameters['b' + str(l)].shape == (self.layers_dims[l], 1))

        return parameters


    def linear_forward(self, A, W, b):
        Z = np.dot(W, A) + b
        
        assert(Z.shape == (W.shape[0], A.shape[1]))
        cache = (A, W, b)
        
        return Z, cache


    def linear_activation_forward(self, A_prev, W, b, activation):
        if activation == "sigmoid":
            Z, linear_cache = self.linear_forward(A_prev, W, b)
            A, activation_cache = sigmoid(Z)
        
        elif activation == "relu":
            Z, linear_cache = self.linear_forward(A_prev, W, b)
            A, activation_cache = relu(Z)

        assert (A.shape == (W.shape[0], A_prev.shape[1]))
        cache = (linear_cache, activation_cache)

        return A, cache


    def L_model_forward(self, X):
        caches = []
        A = X
        L = len(self.parameters) // 2  # number of layers in the neural network
        
        for l in range(1, L):
            A_prev = A 
            A, cache = self.linear_activation_forward(A_prev, 
                                                 self.parameters['W' + str(l)], 
                                                 self.parameters['b' + str(l)], 
                                                 'relu')
            caches.append(cache)
        
        AL, cache = self.linear_activation_forward(A, 
                                              self.parameters['W' + str(L)], 
                                              self.parameters['b' + str(L)], 
                                              self.output_layer)
        caches.append(cache)
        
        assert(AL.shape == (1, X.shape[1]))
                
        return AL, caches


    def compute_cost(self, AL, Y):
        m = Y.shape[0]

        # Compute loss from aL and y.
        cost = -(1/m) * np.sum(Y* np.log(AL) + (1- Y ) * np.log(1- AL))

        # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
        cost = np.squeeze(cost)      
        assert(cost.shape == ())
        
        return cost


    def linear_backward(self, dZ, cache):
        A_prev, W, b = cache
        m = A_prev.shape[1]

        dW = (1/m) * np.dot(dZ, A_prev.T)
        db = (1/m) * np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(W.T, dZ)
        
        assert (dA_prev.shape == A_prev.shape)
        assert (dW.shape == W.shape)
        assert (db.shape == b.shape)
        
        return dA_prev, dW, db


    def linear_activation_backward(self, dA, cache, activation):
        linear_cache, activation_cache = cache
        
        if activation == "relu":
            dZ = relu_backward(dA, activation_cache)
            dA_prev, dW, db = self.linear_backward(dZ, linear_cache)
            
        elif activation == "sigmoid":
            dZ = sigmoid_backward(dA, activation_cache)
            dA_prev, dW, db = self.linear_backward(dZ, linear_cache)

        return dA_prev, dW, db


    def L_model_backward(self, AL, Y, caches):
        grads = {}
        L = len(caches) # the number of layers
        m = AL.shape[1]
        Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL
        
        # Initializing the backpropagation
        dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
        
        # Lth layer (SIGMOID -> LINEAR) gradients. Inputs: "dAL, current_cache". Outputs: "grads["dAL-1"], grads["dWL"], grads["dbL"]
        current_cache = caches[L - 1]
        grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = self.linear_activation_backward(dAL, current_cache, 'sigmoid')
        
        # Loop from l=L-2 to l=0
        for l in reversed(range(L-1)):
            # lth layer: (RELU -> LINEAR) gradients.
            # Inputs: "grads["dA" + str(l + 1)], current_cache". Outputs: "grads["dA" + str(l)] , grads["dW" + str(l + 1)] , grads["db" + str(l + 1)] 
            current_cache = caches[l]
            dA_prev_temp, dW_temp, db_temp = self.linear_activation_backward(grads["dA" + str(l +1)], current_cache, 'relu')
            grads["dA" + str(l)] = dA_prev_temp
            grads["dW" + str(l + 1)] = dW_temp
            grads["db" + str(l + 1)] = db_temp

        return grads


    def update_parameters(self, grads):
        L = len(self.parameters) // 2 # number of layers in the neural network

        # Update rule for each parameter. Use a for loop.
        for l in range(L):
            self.parameters["W" + str(l+1)] -= self.learning_rate * grads["dW" + str(l + 1)]
            self.parameters["b" + str(l+1)] -= self.learning_rate * grads["db" + str(l + 1)]
        return self.parameters


    def fit(self, X, Y):
        # Loop (gradient descent)
        for i in range(0, self.num_iterations):

            # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
            AL, caches = self.L_model_forward(X)
            
            # Compute cost.
            cost = self.compute_cost(AL, Y)
        
            # Backward propagation.
            grads = self.L_model_backward(AL, Y, caches)
     
            # Update parameters.
            self.update_parameters(grads)
                    
            # Print the cost every 10 iterations
            if self.print_cost and i % 10 == 0:
                print ("Cost after iteration %i: %f" %(i, cost))
                
        return self.parameters

    def predict(self, X):
        probas, caches = self.L_model_forward(X)
        
        return probas
