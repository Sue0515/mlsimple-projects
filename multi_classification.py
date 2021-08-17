import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn.preprocessing import LabelEncoder 
from tensorflow.keras.utils import np_utils 

df = pd.read_csv('..\datasets\iris.csv', names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species'])
print(df.describe())

sns.pairplot(df, hue='species')
plt.show() 

dataset = df.values
X = dataset[:, 0:4].astype(float)
Y = dataset[:, 4]

encoder = LabelEncoder() # convert chararray of Y(class names) into number 
encoder.fit(Y)
Y = encoder.transform(Y) # converts ['Iri-setosa', .., .., ] into [1, 2, 3]

# We need to convert Y values (1, 2, 3, .. ) into 0 or 1 in order to pass the value into 
# activation function. Following converts the number into array that only contains 0 and 1 
# [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]
# This is called one hot encoding. 
Y_encoded = tf.keras.utils.to_categorical(Y) 

# configure model  
model = Sequential() 
model.add(Dense(16, input_dim = 5, activation = 'relu'))
model.add(Dense(3, activation = 'softmax'))

# compile model 
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# execute model 
model.fit(X, Y_encoded, epochs = 50, batch_size = 1)

# print result 
print("\n Accuracy: %.4f" % (model.evaluate(X, Y_encoded)[1]))

