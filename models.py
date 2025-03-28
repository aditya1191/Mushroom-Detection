import numpy
import random

from config import LEARNING_RATE
from formulas import sig, inv_sig, inv_err

# Initialize a global variable to keep track of the current node ID
curr_node_id = 0

class Layer:
    def __init__(self, num_nodes, input_vals, layer_num):
        # Initialize a neural network layer with random weights and bias
        self.num_nodes = num_nodes
        self.input_vals = input_vals
        self.layer_num = layer_num
        # Initialize weights with random values
        self.weight = [[random.random() for col in range(len(input_vals))] for row in range(num_nodes)]
        self.weight_delta = [[0 for col in range(len(input_vals))] for row in range(num_nodes)]
        self.layer_net = [0 for col in range(num_nodes)]
        self.layer_out = [0 for col in range(num_nodes)]
        self.bias = (random.random() * 2) - 1

    def eval(self):
        # Evaluate the layer by computing the output of layer nodes using the sigmoid function
        for a in range(self.num_nodes):
            # Compute the net input and output of each node in the layer
            s = self.input_vals
            t = numpy.transpose(self.weight[a])
            s1 = self.bias
            f = numpy.dot(s, t) + s1
            z = sig(f)
            # Save the net input and output for later use
            self.layer_net[a] = f
            self.layer_out[a] = z

    def backprop(self, other):
        # Use backpropagation to update weights based on the gradient descent method
        # Compute the partial derivative of the error with respect to the weights
        k = len(self.weight)
        for i in range(k):
            for j in range(len(self.weight[i])):
                if self.layer_num == 1:
                    # For the second layer, compute the gradient using the chain rule
                    gradient = other.weight_delta[0][i] * self.input_vals[j] * other.weight[0][i] * inv_sig(self.layer_out[i])
                elif self.layer_num == 2:
                    # For the first layer, compute the gradient using the chain rule and error derivatives
                    self.weight_delta[i][j] = inv_sig(self.layer_out[i]) * inv_err(self.layer_out[i], other)
                    gradient = self.weight_delta[i][j] * self.input_vals[j]
                # Update the weight using the learning rate and the computed gradient
                self.weight[i][j] = self.weight[i][j] - (LEARNING_RATE * gradient)

class cfile():
    def __init__(self, name, mode='r'):
        # Initialize a file handler to read or write data
        self.fh = open(name, mode)

    def w(self, string):
        # Write a string to the file
        self.fh.write(str(string) + '\n')
        return None

    def close(self):
        # Close the file
        self.fh.close()
