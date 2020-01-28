# Strategy:
# - Get share price data
# - Calculate technical indicators based on share data
# - Do a sensitivity analysis on the technical indicators to determine which indicators influence the closing price the most
# - Use most important data to train the neural network
# - Train the network and evaluate model

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import Sequential
from keras.layers import Flatten
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import math
import sensitivity_analysis as sa
import technical_indicator_matrix as tim

# This is done to ensure the results are reproducible
np.random.seed(7)

# Read data from file
data = pd.read_csv('TestData/HananData/BAB_LN.csv')

# Declare high, low, close, volume variables
high = np.array(data['HIGH'].values)
low = np.array(data['LOW'].values)
close = np.array(data['LAST_PRICE'].values)
volume = np.array(data['VOLUME'].values)

# Sliding window
period = 20

# Construct matrix with technical indicators (SMA, EWMA, Bollinger Bands, CCI, ROC, FI, EVM)
# Library used for technical indicators: https://github.com/kylejusticemagnuson/pyti
technical_indicator_names = ['CCI', 'SMA', 'EWMA', 'ROC', 'LBB', 'UBB', 'FI', 'EVM']
technical_indicator_matrix = np.array(tim.create_simple_matrix(high, low, close, volume, period))

matrix_placeholders = []

event_threshold = 0.05
trigger_threshold = 0.05

# Discard first 19 values which is 'nan' values
technical_indicator_matrix = technical_indicator_matrix[0:technical_indicator_matrix.shape[0], period-1:technical_indicator_matrix.shape[1]-1]

# Sensitivity analysis to determine technical indicators' influence on closing price
sensitivity_matrix = sa.analyse_sensitivity(trigger_data=technical_indicator_matrix, event_data=close, trigger_threshold=trigger_threshold, event_threshold=event_threshold)

# Filter data based on sensitivity analysis
filtered_data = []
sensitivity_analysis_values = []
filtered_technical_indicator_names = []
for i, col in enumerate(sensitivity_matrix):
    average_sensitivity = pd.Series(col).mean()
    if average_sensitivity > 0.5:
        filtered_data.append(technical_indicator_matrix[i])
        matrix_placeholders.append(i)
        sensitivity_analysis_values.append(average_sensitivity)
        filtered_technical_indicator_names.append(technical_indicator_names[i])
filtered_data = np.array(filtered_data)

# Filtered technical indicators after sensitivity analysis was completed
print(filtered_technical_indicator_names)
print(sensitivity_analysis_values)

# Split data into training and testing data sets
train_size = int(filtered_data.shape[1] * 0.67)
test_size = int(filtered_data.shape[1] - train_size)
train_data = filtered_data[0:filtered_data.shape[0], 0:train_size]
test_data = filtered_data[0:filtered_data.shape[0], train_size:filtered_data.shape[1]]

# Closing training and testing data
close = close[period:close.shape[0]]
train_close = close[0:train_size]
test_close = close[train_size:filtered_data.shape[1]]

# Normalize data
test_scaler = MinMaxScaler()
train_scaler = MinMaxScaler()
train_data = np.vstack([train_data, train_close])
test_data = np.vstack([test_data, test_close])
train_data = train_scaler.fit_transform(train_data)
test_data = test_scaler.fit_transform(test_data)

train_close = train_data[train_data.shape[0]-1:train_data.shape[0], :][0]
test_close = test_data[test_data.shape[0]-1:test_data.shape[0], :][0]
train_data = train_data[0:train_data.shape[0]-1, :]
test_data = test_data[0:test_data.shape[0]-1, :]

# reshape data set
train_data = np.reshape(train_data, (train_data.shape[1], 1, train_data.shape[0]))
test_data = np.reshape(test_data, (test_data.shape[1], 1, test_data.shape[0]))

# Construct and train neural network
# Model contains 3 hidden layers with 64 neurons each
# Library documentation can be found at http://www.keras.io
model = Sequential()
model.add(LSTM(4, input_shape=(train_data.shape[1], train_data.shape[2]), return_sequences=True))
model.add(Dense(64))
model.add(Dense(64))
model.add(Dense(64))
model.add(Flatten())
model.add(Dense(units=1))
# model.compile(loss='mean_squared_error', optimizer='Adadelta', metrics=['accuracy'])
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
model.fit(train_data, train_close, epochs=100, batch_size=60, verbose=2)
loss, accuracy = model.evaluate(test_data, test_close)
print("\nLoss: %.5f, Accuracy: %.5f%%" % (loss, accuracy * 100))

train_predict = model.predict(train_data)
test_predict = model.predict(test_data)

# Invert predictions
train_data = np.reshape(train_data, (train_data.shape[2], train_data.shape[0]))
test_data = np.reshape(test_data, (test_data.shape[2], test_data.shape[0]))

train_data = np.vstack([train_data, train_close])
train_data = np.vstack([train_data, np.reshape(train_predict, (np.array(train_predict).shape[0], ))])
train_data = train_scaler.inverse_transform(train_data)

test_data = np.vstack([test_data, test_close])
test_data = np.vstack([test_data, np.reshape(test_predict, (np.array(test_predict).shape[0], ))])
test_data = test_scaler.inverse_transform(test_data)

# Plot predicted values and actual values
plt.figure(1)
plt.plot(test_data[test_data.shape[0]-1], 'b-', test_data[test_data.shape[0]-2], 'y-')
plt.xlabel("blue - predicted, yellow - actual")
plt.figure(2)
plt.plot(train_data[train_data.shape[0]-1], 'b-', train_data[train_data.shape[0]-2], 'y-')
plt.xlabel("blue - predicted, yellow - actual")
plt.show()
