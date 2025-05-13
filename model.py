import numpy as np
import matplotlib.pyplot as plt
import copy

class myNN :
    def __init__(self , layer_dims , learning_rate, seed):
        np.random.seed(seed)
        self.layer_dims = layer_dims
        self.learning_rate = learning_rate
        self.parameters = self.initialize_parameters()

    def initialize_parameters(self) :
        parameters = {}
        L = len(self.layer_dims)
        for i in range(1,L):
            parameters["W" + str(i)] = np.random.randn(self.layer_dims[i], self.layer_dims[i-1]) * np.sqrt(2 / self.layer_dims[i-1])
            parameters["b" + str(i)] = np.zeros((self.layer_dims[i] , 1))
        return parameters
    
    def softMax(self ,Z):
        Z_shift = Z - np.max(Z, axis=0, keepdims=True)
        exp_Z = np.exp(Z_shift)
        return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)

    def linear_activation_forward(self , A_prev , W , b , activation) :
        if(activation == "relu") :
            Z = np.dot(W , A_prev) + b
            linear_cache = (A_prev , W , b)
            A = np.maximum(0 , Z)
            activation_cache = Z
        elif(activation == "softMax") :
            Z = np.dot(W , A_prev) + b
            linear_cache = (A_prev , W , b)
            A = self.softMax(Z)
            activation_cache = Z
        elif(activation == "sigmoid") :
            Z = np.dot(W, A_prev) + b
            linear_cache = (A_prev, W, b)
            A = 1 / (1 + np.exp(-Z))
            activation_cache = Z

        cache = (linear_cache , activation_cache)
        return A , cache
    
    def L_forwarPropagation(self , X) :
        caches = []
        A = X
        L = len(self.parameters) // 2

        for l in range(1 , L) :
            A_prev = A
            A , cache = self.linear_activation_forward(A_prev , self.parameters["W" + str(l)] , self.parameters["b" + str(l)] , "relu")
            caches.append(cache)
        
        AL , cache = self.linear_activation_forward(A , self.parameters["W" + str(L)] , self.parameters["b" + str(L)] , "softMax")
        caches.append(cache)

        return AL , caches
    
    def compute_cost(self , AL , Y) :
        m = Y.shape[1]
        cost = -1/m*(np.sum(Y * np.log(AL + 1e-8)))
        return np.squeeze(cost)
    
    def linear_backward(self , dZ , cache) :
        A_prev , W , b = cache
        m = A_prev.shape[1]

        dW = 1/m * np.dot(dZ , A_prev.T)
        db = 1/m * (np.sum(dZ , axis=1 , keepdims=True))
        dA_prev = np.dot(W.T , dZ)

        return dA_prev , dW , db
    
    def linear_activation_backward(self, dA, cache, activation, Y=None):
        linear_cache, activation_cache = cache

        if activation == "relu":
            Z = activation_cache
            dZ = np.array(dA, copy=True)
            dZ[Z <= 0] = 0
            dA_prev, dW, db = self.linear_backward(dZ, linear_cache)

        elif activation == "sigmoid":
            Z = activation_cache
            s = 1 / (1 + np.exp(-Z))
            dZ = dA * s * (1 - s)
            dA_prev, dW, db = self.linear_backward(dZ, linear_cache)

        elif activation == "softmax":
            A = activation_cache  # notice: softmax caches A, not Z
            dZ = A - Y  # softmax + cross-entropy combined derivative
            dA_prev, dW, db = self.linear_backward(dZ, linear_cache)

        return dA_prev, dW, db
    
    def L_backwardPropagation(self , AL , Y , caches) :
        grads = {}
        L = len(caches)
        m = AL.shape[1]
        Y = Y.reshape(AL.shape)

        current_cache = caches[L-1]
        dA_prev_temp , dW_temp , db_temp = self.linear_activation_backward(None , current_cache , "softmax" , Y)
        grads["dA" + str(L-1)] = dA_prev_temp
        grads["dW" + str(L)] = dW_temp
        grads["db" + str(L)] = db_temp

        for l in reversed(range(L-1)) : 
            dA_prev_temp , dW_temp , db_temp = self.linear_activation_backward(grads["dA" + str(l+1)] , caches[l] , "relu" )
            grads["dA" + str(l)] = dA_prev_temp
            grads["dW" + str(l+1)] = dW_temp
            grads["db" + str(l+1)] = db_temp
        
        return grads
    
    def clip_gradients(self, grads, max_norm=0.1):
        total_norm = np.sqrt(sum(np.sum(g ** 2) for g in grads.values()))
        if total_norm > max_norm:
            for key in grads.keys():
                grads[key] = grads[key] * (max_norm / (total_norm + 1e-6))
        return grads


    def update_parameters(self , parameters , grads) :
        grads = self.clip_gradients(grads)
        L = len(parameters) // 2

        for l in range(0 , L) :
            parameters["W" + str(l+1)] -= self.learning_rate * grads["dW" + str(l+1)]
            parameters["b" + str(l+1)] -= self.learning_rate * grads["db" + str(l+1)]

        return parameters

    def train_batch(self , X_batch , Y_batch , num_iterations=1) :
        np.random.seed(1)
        costs = [] 

        parameters = self.initialize_parameters()

        for i in range(num_iterations) :
            AL , caches = self.L_forwarPropagation(X_batch)
            cost = self.compute_cost(AL , Y_batch)
            grads = self.L_backwardPropagation(AL , Y_batch , caches)
            self.parameters = self.update_parameters(parameters , grads)
        return cost                                                                                             

    def predict(self , X) : 
        AL , _ = self.L_forwarPropagation(X)
        predictions = np.argmax(AL , axis = 0)
        return predictions
    
    