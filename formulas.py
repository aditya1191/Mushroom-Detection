import math

def sig(x):
    # Use the logistic function as the activation function
    # The sigmoid function is defined as 1 / (1 + e^(-x))
    den = math.e**(-x)
    return 1 / (1.0 * (1 + den))

def inv_sig(x):
    # Calculate the derivative of the neuron's output with respect to its input
    # The derivative of the sigmoid function sig'(x) is given by sig(x) * (1 - sig(x))
    y = sig(x)
    return y * (1 - y)

def err(o, t):
    # Compute the squared error between the actual output (o) and the target output (t)
    # The squared error function is defined as 1/2 * (target - output)^2
    return (0.5 * (math.pow(t - o, 2)))

def inv_err(o, t):
    # Calculate the derivative of the squared error function with respect to the actual output (o)
    # The derivative of the squared error function err'(o, t) with respect to o is (output - target)
    return (o - t)
