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
from keras.models import Sequential
import sensitivity_analysis as sa
import technical_indicator_matrix as tim

# Read data from files (train, test & validation)
train_data = pd.read_csv('BP1.csv')
test_data = pd.read_csv('Anglo_American_plc.csv')
validate_data = pd.read_csv('Oil_Gas_producer_Class_B.csv')

matrix_placeholders = []

# Sliding window
period = 4

# Declare train data sets - high, low, close, volume
# Dates,Open,Close,High,Low,Value,Volume
train_high = np.array(train_data['Dates'].values)
train_low = np.array(train_data['Open'].values)
train_close = np.array(train_data['Close'].values)
train_high = np.array(train_data['High'].values)
train_low = np.array(train_data['Low'].values)
train_value = np.array(train_data['Value'].values)
train_volume = np.array(train_data['Volume'].values)

# Declare test data sets - high, low, close, volume
#test_high = np.array(test_data['HIGH'].values)
#test_low = np.array(test_data['LOW'].values)
#test_close = np.array(test_data['LAST_PRICE'].values)
#test_volume = np.array(test_data['VOLUME'].values)

test_high = np.array(test_data['Dates'].values)
test_low = np.array(test_data['Open'].values)
test_close = np.array(test_data['Close'].values)
test_high = np.array(test_data['High'].values)
test_low = np.array(test_data['Low'].values)
test_value = np.array(test_data['Value'].values)
test_volume = np.array(test_data['Volume'].values)


# Declare validation data sets - high, low, close, volume
#validate_high = np.array(validate_data['HIGH'].values)
#validate_low = np.array(validate_data['LOW'].values)
#validate_close = np.array(validate_data['LAST_PRICE'].values)
#validate_volume = np.array(validate_data['VOLUME'].values)

validate_high = np.array(validate_data['Dates'].values)
validate_low = np.array(validate_data['Open'].values)
validate_close = np.array(validate_data['Close'].values)
validate_high = np.array(validate_data['High'].values)
validate_low = np.array(validate_data['Low'].values)
validate_value = np.array(validate_data['Value'].values)
validate_volume = np.array(validate_data['Volume'].values)

# Construct matrices with technical indicators (SMA, EWMA, Bollinger Bands, CCI, ROC, FI, EVM)
# Library used for technical indicators: https://github.com/kylejusticemagnuson/pyti
technical_indicator_names = ['CCI', 'SMA', 'EWMA', 'ROC', 'LBB', 'UBB', 'FI', 'EVM']
ti_train_matrix = np.array(tim.create_simple_matrix(train_high, train_low, train_close, train_volume, period))
#ti_test_matrix = np.array(tim.create_simple_matrix(test_high, test_low, test_close, test_volume, period))
ti_test_matrix = np.array(tim.create_simple_matrix(test_high, test_low, test_close, test_volume, period))
#ti_validate_matrix = np.array(tim.create_simple_matrix(validate_high, validate_low, validate_close, validate_volume, period))
ti_validate_matrix = np.array(tim.create_simple_matrix(validate_high, validate_low, validate_close, validate_volume, period))

event_threshold = 0.05
trigger_threshold = 0.05

# Replace all 'nan' values with 0
ti_train_matrix[np.isnan(ti_train_matrix)] = 0

# Sensitivity analysis to determine technical indicators' influence on closing price
sensitivity_matrix = sa.analyse_sensitivity(trigger_data=ti_train_matrix, event_data=train_close, trigger_threshold=trigger_threshold, event_threshold=event_threshold)


# Filter training data based on sensitivity analysis
data_to_train = []
sensitivity_analysis_values = []
for i, col in enumerate(sensitivity_matrix):
    average_sensitivity = pd.Series(col).mean()
    if average_sensitivity > 0.5:
        data_to_train.append(ti_train_matrix[i])
        matrix_placeholders.append(i)
        sensitivity_analysis_values.append(average_sensitivity)
data_to_train = np.array(data_to_train)

# Construct testing and validation matrices based on sensitivity analysis on training data
data_to_test = []
data_to_validate = []
filtered_technical_indicator_names = []
for i, col in enumerate(matrix_placeholders):
    data_to_test.append(ti_test_matrix[col])
    data_to_validate.append(ti_validate_matrix[col])
    filtered_technical_indicator_names.append(technical_indicator_names[col])
data_to_test = np.array(data_to_test)
print("data_to_test %s", data_to_test)
data_to_validate = np.array(data_to_validate)
print("data_to_validate %s",data_to_validate)

# Filtered technical indicators after sensitivity analysis was completed
print(filtered_technical_indicator_names)
print(sensitivity_analysis_values)

# Replace all 'nan' values with 0
data_to_train[np.isnan(data_to_train)] = 0
data_to_test[np.isnan(data_to_test)] = 0
data_to_validate[np.isnan(data_to_validate)] = 0

# Construct and train neural network
# Model contains 3 hidden layers with 64 neurons each
# Library documentation can be found at http://www.keras.io
model = Sequential()
model.add(Dense(units=64, activation='linear', input_shape=(data_to_train.shape[0],)))
model.add(Dense(units=64, activation='linear'))
model.add(Dense(units=64, activation='linear'))
model.add(Dense(units=64, activation='linear'))
model.add(Dense(units=1, activation='linear'))
model.compile(loss='mean_squared_error', optimizer='Adadelta', metrics=['accuracy'])
history = model.fit(data_to_train.T, train_close,  epochs=50)
#history = model.fit(data_to_train.T, train_close, validation_data=([data_to_validate.T, validate_close]), epochs=50)
loss, accuracy = model.evaluate(data_to_validate.T, validate_close)
print("\nLoss: %.2f, Accuracy: %.2f%%" % (loss, accuracy * 100))

# Evaluate model with test data
test_date_data = test_data['Dates'].values

# Plot predicted values and actual values
plt.figure(1)
plt.plot(test_date_data, model.predict(data_to_test.T), 'r-', test_date_data, test_close, 'b-')
plt.show()
