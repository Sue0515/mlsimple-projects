import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 

data = [[1.5, 0], [2, 0], [2.5, 0], 
        [6, 1], [7, 1], [10,1]]

X = [i[0] for i in data]
Y = [i[1] for i in data]

plt.scatter(X, Y)
plt.xlim(0, 11)
plt.ylim(-.1, 1.1)

a = 0
b = 0
epoch = 2001

learning_rate = 0.03

def sigmoid_func(x):
    return 1 / (1 + np.e**(-x))

for i in range(epoch):
    for X, Y in data:
        a_diff = X * (sigmoid_func(a * X + b) - Y)
        b_diff = sigmoid_func(a * X + b) - Y
        a = a - learning_rate * a_diff
        b = b - learning_rate * b_diff 

x_range = np.arange(0, 11, 0.1)
plt.plot(x_range, np.array([sigmoid_func(a * x + b) for x in x_range]))
plt.show()
    

