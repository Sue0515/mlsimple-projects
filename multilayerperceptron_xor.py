import numpy as np

input = [(0, 0), (1, 0), (0, 1), (1, 1)]
w1121 = np.array([-2, -2])
w1222 = np.array([2, 2])
w3132 = np.array([1, 1])
b1 = 3
b2 = -1
b3 = -1

def multi_layer_perceptron(x, w, b):
    node = np.sum(w * x) + b
    if node > 0:
        return 1
    else:
        return 0

def nand_gate(x, y):
    return multi_layer_perceptron(np.array([x, y]), w1121, b1)

def or_gate(x, y):
    return multi_layer_perceptron(np.array([x, y]), w1222, b2)

def and_gate(x, y):
    return multi_layer_perceptron(np.array([x, y]), w3132, b3)

def xor_gate(x, y):
    return and_gate(nand_gate(x, y), or_gate(x, y))

for i in input:
    result = xor_gate(i[0], i[1])
    print("input: " + str(i) + " => output: " + str(result))
    
