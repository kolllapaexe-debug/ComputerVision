import pandas as pd #csv таблиці
import numpy as np #math
import tensorflow as tf #neiromerezha
from tensorflow import keras
 #wide tensorflow
from tensorflow.keras import layers #wider for layers
from sklearn.preprocessing import LabelEncoder #text to numbers
import matplotlib.pyplot as plt #
#2 csv files
df = pd.read_csv('data/figures.csv')
print(df.head())
#3 name in numbers
encoder = LabelEncoder()
df['Label_enc'] = encoder.fit_transform(df['label'])
# tables for learning
X = df[["Area", "Perimeter", "Corners"]]
y = df["Label_enc"]
#4 creating model
model = keras.Sequential([
    layers.Dense(8, activation='relu', input_shape= (3,)),
    layers.Dense(8, activation='relu'),
    layers.Dense(8, activation='softmax'),
])
#6 compilation of model
model.compile(optimizer='adam',loss= 'sparse_categorical_crossentropy',metrics=['accuracy'])
history = model.fit(X, y, epochs= 300, verbose= 0)
plt.plot(history.history['loss'], label = 'loss')
plt.plot(history.history['accuracy'], label = 'accuracy')
plt.xlabel('epoch')
plt.ylabel('value')
plt.title("learning")
plt.legend()
plt.show()
#7 testing
test = np.array([25,20,0])
pred = model.predict(test)
print(pred)