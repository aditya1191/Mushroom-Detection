import math

def sig(x):
    #use logistic function as activation function
    #Sigmoid function = 1/(1+e^(-x))
    den = math.e**(-x)
    return 1/(1.0*(1+den))

def inv_sig(x):
    #derivative of the output of neruon with respect to its input
    #sig'(x) = e^(-x)/((1+e^(-x))^2)
    #sig'(x) = sig(x) * (1-sig(x))
    y = sig(x)
    return y*(1-y)

def err(o, t):
    #squared error function, o is the actual output value and t is the target output
    # 1/2 ((target - output)^2)
    return (0.5 * (math.pow(t-o,2)))

def inv_err(o, t):
    #derivative of squared error function with respect to o
    #err'(o,t) w.r.t o = 0.5 * 2 * (target - output) = (output - target)
    return (o-t)


