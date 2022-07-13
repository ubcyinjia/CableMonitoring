# Load libraries
import pandas
import numpy
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt

# Load dataset
X_train = pandas.read_csv('X_train.csv',header=None);
y_train = pandas.read_csv('y_train.csv',header=None);

X_test = pandas.read_csv('X_test.csv',header=None);
y_test = pandas.read_csv('y_test.csv',header=None);

# Construct Neural Network
import keras

from keras import models
from keras import layers
from keras.layers.normalization import BatchNormalization
from keras import regularizers

X_train = numpy.array(X_train)
y_train = numpy.array(y_train)
X_test = numpy.array(X_test)
y_test = numpy.array(y_test)

X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
y_train = y_train.reshape(y_train.shape[0], 1, y_train.shape[1])
X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
y_test = y_test.reshape(y_test.shape[0], 1, y_test.shape[1])


model = models.Sequential();
model.add(layers.LSTM(8, input_shape=(1,X_train.shape[2]), return_sequences=True))
#model.add(layers.LSTM(16, input_shape=(1,X_train.shape[2]), return_sequences=True))
#model.add(layers.LSTM(16))
#model.add(layers.Dense(256, activation='tanh', input_shape=(1080,)))
#model.add(BatchNormalization())
#model.add(layers.Dense(64, activation='sigmoid'))
#model.add(BatchNormalization())
#model.add(layers.Dense(16, activation='relu'))
#model.add(BatchNormalization())
#model.add(layers.Dense(8, activation='relu'))
model.add(layers.Dense(1, activation='linear'))


#model.summary()

from keras import optimizers
model.compile(optimizer='rmsprop', loss='mse', metrics=['mse'])

# Train neural network
history = model.fit(X_train, y_train, batch_size=256, epochs=50, validation_data=(X_test, y_test))
model.save('test_neural.h5')


# summarize history for accuracy
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model MSE loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('result')

# Making the prediction
y_predict_train = model.predict(X_train)
y_predict_test = model.predict(X_test)

y_predict_train = y_predict_train.reshape(y_predict_train.shape[0], 1)
y_predict_test = y_predict_test.reshape(y_predict_test.shape[0], 1)
#pandas.DataFrame(y_predict_train).to_csv("s_matrix_predict_train.csv", header=None, index=None)
pandas.DataFrame(y_predict_test).to_csv("snr_prediction.csv", header=None, index=None)


#automl.fit(X_train, y_train);
#y_hat = automl.predict(X_test);
#print("R2 score:", sklearn.metrics.r2_score(y_test, y_hat));

#import numpy;
#numpy.savetxt("out.csv",y_hat,delimiter=",");

#text_file = open("model.txt", "w");
#text_file.write(automl.show_models());
#text_file.close();
