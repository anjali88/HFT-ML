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

import xlrd
import csv
import sys

csv_file_list=[]
period=600
event_threshold = 0.01
trigger_threshold = 0.01

excel_sheet_name=sys.argv[1]
sheets=pd.ExcelFile(excel_sheet).sheet_names
train_data={}
for sheet in sheets:
    _train_data=pd.excel_file(excel_sheet,sheet_name=sheet)
    sheet_name=sheet.replace(' ','_')
    sheet_name=sheet.replace('&','')
    train_data[sheet_name]=_train_data
    




    








with xlrd.open_workbook(sys.argv[1]) as wb:
    sheets=wb.sheet_names()
    for sheet in sheets:
        sheet1=sheet
        sheet=wb.sheet_by_name(sheet)
        sheet1=sheet1.replace(' ','_')
        sheet1=sheet1.replace('&','_')
        sheet_name="{}.csv".format(sheet1)
        csv_file_list.append(sheet_name)
        with open(sheet_name, 'w') as f:   # open('a_file.csv', 'w', newline="") for python 3
            c = csv.writer(f)
            for r in range(sheet.nrows):
                #print(sheet.row_values(r))
                c.writerow(sheet.row_values(r))

# Read data from files (train, test & validation)
train_data={}
for csv_file in csv_file_list:
    sheet_name=csv_file[:-4]
    if sheet_name not in train_data.keys():
        train_data[sheet_name]={}
    train_data[sheet_name]=pd.read_csv(csv_file)

train_data_dict={}
for _train_data in train_data.keys():
    if _train_data not in train_data_dict.keys():
        train_data_dict[_train_data]={}
    train_data_dict[_train_data]["dates"]=np.array(train_data[_train_data]['Dates'].values)
    train_data_dict[_train_data]["high"]=np.array(train_data[_train_data]['High'].values)
    train_data_dict[_train_data]["low"]=np.array(train_data[_train_data]['Low'].values)
    train_data_dict[_train_data]["open"]=np.array(train_data[_train_data]['Open'].values)
    train_data_dict[_train_data]["close"]=np.array(train_data[_train_data]['Close'].values)
    train_data_dict[_train_data]["value"]=np.array(train_data[_train_data]['Value'].values)
    train_data_dict[_train_data]["volume"]=np.array(train_data[_train_data]['Volume'].values)

technical_indicator_names = train_data.keys()
train_data_matrix={}

for _train_data in train_data.keys():
    if _train_data not in train_data_matrix.keys():
        train_data_matrix[_train_data]={}
    train_data_matrix[_train_data]={}
    train_data_matrix[_train_data]["close"]=np.array(train_data_dict[_train_data]["close"])
    mean=np.array(train_data_dict[_train_data]["close"]).mean()
    std=np.array(train_data_dict[_train_data]["close"]).std()
    train_data_matrix[_train_data]["lower"]=mean-std/2
    train_data_matrix[_train_data]["upper"]=mean+std/2
    train_data_matrix[_train_data]["name"]=_train_data
    train_data_matrix[_train_data]["close"][np.isnan(train_data_matrix[_train_data]["close"])]=0    

train_data_input=[t_data for t_data in train_data.keys()]
train_data_input=train_data_input[1:]
train_data_output=[t_data for t_data in train_data.keys()]
train_data_output=train_data_output[0]
ti_train_matrix={}
for _trigger in train_data_input:
    ti_train_matrix[_trigger]=train_data_matrix[_trigger]

#ti_train_matrix=[train_data_matrix[_train_data] for _train_data in train_data_input]
train_analyse_matrix = sa.analyse_sensitivity_multiple_input(trigger_data=ti_train_matrix, event_data=train_data_matrix[train_data_output]["close"], trigger_threshold=trigger_threshold, event_threshold=event_threshold,period=period)

matrix_placeholders = []

# Sliding window

# Sensitivity analysis to determine technical indicators' influence on closing price
#sensitivity_matrix = sa.analyse_sensitivity(trigger_data=ti_train_matrix, event_data=train_close, trigger_threshold=trigger_threshold, event_threshold=event_threshold)


# Filter training data based on sensitivity analysis
data_to_train = []
sensitivity_analysis_values = []

for _trigger in train_analyse_matrix.keys():
    normal_si=train_analyse_matrix[_trigger]["normal_si"]
    average_sensitivity =np.array(normal_si).mean()
    avg=np.array(normal_si).mean()
    std=np.array(normal_si).std()
    lower=avg-std
    upper=avg+std
    train_analyse_matrix[_trigger]["filter"]=[]
    for i in normal_si:
        if i >= lower and i <= upper:
            train_analyse_matrix[_trigger]["filter"].append(i)
    train_analyse_matrix[_trigger]["filter"]=np.array(train_analyse_matrix[_trigger]["filter"])
    print(train_analyse_matrix[_trigger]["filter"])

plt.title("Figure: Event Tracker Filtered data")
lables=[]
for _trigger in train_analyse_matrix.keys():
    lables.append(_trigger)
    plt.plot(np.array(train_analyse_matrix[_trigger]["filter"]))
plt.legend(lables)    
plt.savefig('event_tracker_filter.png')
plt.close()

plt.title("Figure: Sensitivity Index")
lables=[]
for _trigger in train_analyse_matrix.keys():
    lables.append(_trigger)
    plt.plot(np.array(train_analyse_matrix[_trigger]["si"]))
plt.legend(lables)    
plt.savefig('sensitivity_index.png')
plt.close()

plt.title("Figure: Normalized Sensitive Index")
lables=[]
for _trigger in train_analyse_matrix.keys():
    lables.append(_trigger)
    plt.plot(np.array(train_analyse_matrix[_trigger]["normal_si"]))
plt.legend(lables)    
plt.savefig('normal_sensitive_index.png')
plt.close()


# Construct and train neural network
# Model contains 3 hidden layers with 64 neurons each
# Library documentation can be found at http://www.keras.io
model = Sequential()
model.add(Dense(units=64, activation='linear', input_shape=(train_analyse_matrix[train_data_output]["normal_si"],)))
model.add(Dense(units=64, activation='linear'))
model.add(Dense(units=64, activation='linear'))
model.add(Dense(units=64, activation='linear'))
model.add(Dense(units=1, activation='linear'))
model.compile(loss='mean_squared_error', optimizer='Adadelta', metrics=['accuracy'])
history = model.fit(train_analyse_matrix[train_data_output]["normal_si"], train_analyse_matrix[train_data_output]["normal_si"], validation_data=([data_to_validate.T, validate_close]), epochs=50)
loss, accuracy = model.evaluate(data_to_validate.T, validate_close)
print("\nLoss: %.2f, Accuracy: %.2f%%" % (loss, accuracy * 100))

# Evaluate model with test data
test_date_data = test_data['Dates'].values

# Plot predicted values and actual values
plt.figure(1)
plt.plot(test_date_data, model.predict(data_to_test.T), 'r-', test_date_data, test_close, 'b-')
plt.show()













