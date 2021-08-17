import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import tensorflow as tf 

df = pd.read_csv('..\datasets\pima-indians-diabetes.csv',
                    names = ['pregnant', 'plasma', 'pressure', 'thickness',
                    'insulin', 'BMI', 'pedigree', 'age', 'class'])

print(df.describe())
print(df[['BMI', 'class']])
print(df[['pregnant', 'class']].groupby(['pregnant'], as_index=False).mean().sort_values(by='pregnant', ascending=True)

plt.figure(figsize = (12,12)) # determine the size of the figure 
sns.heatmap(df.corr(), linewidths = 0.1, vmax = 0.5, cmap = plt.cm.gist_heat, linecolor = 'white', annot = True)
plt.show()

grid = sns.FacetGrid(df, col = 'class')
grid.map(plt.hist, 'plasma', bins=10)
plt.show() 

############DL

numpy.random.seed(3)
tf.random.set_seed(3)

dataset = numpy.loadtxt('..\datasets\pima-indians-diabetes.csv', delimiter=',')
X = dataset[:, 0:8]
Y = dataset[:, 8]

model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer = 'adam', metrics=['accuracy'])

model.fit(X, Y, epochs = 100, batch_size = 10)
print("\n Acurracy: %.4f" % (model.evaluate(X, Y)[1]))
