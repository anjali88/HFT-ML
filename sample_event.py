# Strategy:
# - Get share price data
# - Calculate technical indicators based on share data
# - Do a sensitivity analysis on the technical indicators to determine which indicators influence the closing price the most
# - Use most important data to train the neural network
# - Train the network and evaluate model

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
from keras.layers import Dense
from keras.models import Sequential
import sensitivity_analysis_multi as sa
import technical_indicator_matrix as tim
import seaborn as sns
import xlrd
import csv
import sys
from support_vector_regression import svm_svr
from keras_reg import linear_reg
from arima_analysys import arima_prediction

csv_file_list=[]
period=1
event_threshold = 0.001
trigger_threshold = 0.001



excel_sheet=sys.argv[1]
sheets=pd.ExcelFile(excel_sheet).sheet_names
#print("Sheets {}".format(sheets))

train_data={}
min_length=[]
for sheet in sheets:
    _train_data=pd.read_excel(excel_sheet,sheet_name=sheet)
    s_name=sheet.replace(' ','_')
    s_name=s_name.replace('&','_')
    train_data[s_name]=_train_data
    min_length.append(len(_train_data))

min_rows=min(min_length)

if len(min_length) > 1:
    for _company in train_data.keys():
        train_data[_company]=train_data[_company][:min_rows]

"""
while True:
    i=1
    for company in train_data.keys():
        print("{}){}".format(i,company))
        i+=1
    input_companies=input("Select the Input Company list more than one")
    i=1
    print("===========================================================")
    for company in train_data.keys():
        print("{}){}".format(i,company))
        i+=1
    out_companies=input("Select the OutPut Company(Only one)")
    if len(out_companies) ==1 and out_companies not in input_companies:
        break
print("input company:{} output company:{}".format(input_companies,out_companies))
input_train_data={}
output_train_data={}
print("{}".format(train_data.keys()))
for in_com in input_companies:
    company=list(train_data.keys())[int(in_com)-1]
    input_train_data[company]=train_data[company]
out_company=list(train_data.keys())[int(out_companies)-1]
output_train_data[out_company]=train_data[out_company]
"""
input_train_data={}
output_train_data={}

validate_input_matrix=pd.DataFrame()
validate_output_matrix=pd.DataFrame()

input_company=list(train_data.keys())[1:]
for company in input_company:
    input_train_data[company]=train_data[company]
    validate_input_matrix=validate_input_matrix.assign(**{company:train_data[company]["Close"]})
out_company=list(train_data.keys())[0]
output_train_data=train_data[out_company]
validate_output_matrix=validate_output_matrix.assign(**{out_company:train_data[out_company]["Close"]})

input_test_data=pd.DataFrame()


# GENERATING CORELATION MATRIX
columns={}
for _trigger in list(input_train_data.keys()):
     columns[_trigger]=input_train_data[_trigger]["Close"]
out_comp=out_company
columns[out_comp]=output_train_data["Close"]
data_to_train=pd.DataFrame(columns)

cor=data_to_train.corr(method='pearson')
cor=cor.drop(columns=input_company)
cor=cor.drop(out_comp)

plt.title("Figure: Corelation Matrix Without Event Tracker")
cm=plt.cm.viridis
svm = sns.heatmap(cor,cmap=cm,linewidths=0.1,linecolor='white',annot=True)
plt.savefig('correlation_matrix_without_event_tracker.png')
plt.close()

# SENSITIVITY ANALYSYS
data_input_dataframe=pd.DataFrame()
data_output_dataframe=pd.DataFrame()

train_analyse_matrix, event_analyse_matrix = sa.analyse_sensitivity_multiple_input(trigger_data=input_train_data, event_data=output_train_data, trigger_threshold=trigger_threshold, event_threshold=event_threshold,period=period)

data_to_train=pd.DataFrame()
columns={}
for _trigger in list(train_analyse_matrix.keys()):
     columns[_trigger]=train_analyse_matrix[_trigger]["Close"]
data_input_dataframe=pd.DataFrame(columns)
out_comp=out_company
columns[out_comp]=event_analyse_matrix["Close"]
#print(event_analyse_matrix)
data_output_dataframe=data_output_dataframe.assign(**{out_comp:event_analyse_matrix["Close"]})
data_to_train=pd.DataFrame(columns)

#print(data_to_train.head())

validate_data_matrix=output_train_data
train_event_data=event_analyse_matrix

data_to_train = []
sensitivity_analysis_values = []


plt.title("Figure: Sensitivity Index")
lables=[]
for _trigger in train_analyse_matrix.keys():
    lables.append(_trigger)
    plt.plot(np.array(train_analyse_matrix[_trigger]["s_index"]))
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

for _trigger in train_analyse_matrix.keys():
    normal_si=train_analyse_matrix[_trigger]["normal_si"].tolist()
    average_sensitivity =np.array(normal_si).mean()
    avg=np.array(normal_si).mean()
    std=np.array(normal_si).std()
    lower=avg-std
    upper=avg+std
    for i in train_analyse_matrix[_trigger].index.tolist():
        value=train_analyse_matrix[_trigger]["normal_si"][i]
        if value < lower or value > upper:
            train_analyse_matrix[_trigger]=train_analyse_matrix[_trigger].drop(i)
            train_event_data=train_event_data.drop(i)

train_event_data=train_event_data.fillna(0)

plt.title("Figure: Filtered Normalized Sensitive Index")
lables=[]
for _trigger in train_analyse_matrix.keys():
    lables.append(_trigger)
    plt.plot(np.array(train_analyse_matrix[_trigger]["normal_si"]))
plt.legend(lables)
plt.savefig('filtered_normal_sensitive_index.png')
plt.close()


plt.title("Figure: Filtered Sensitivity Index")
lables=[]
for _trigger in train_analyse_matrix.keys():
    lables.append(_trigger)
    plt.plot(np.array(train_analyse_matrix[_trigger]["s_index"]))
plt.legend(lables)
plt.savefig('filtered_sensitivity_index.png')
plt.close()


avg_event=[]
for _trigger in train_analyse_matrix.keys():
    avg_event.append(average_sensitivity)

plt.title("Figure: Averaged Normalised Sensitivity Index WRT {} TDS".format(len(train_analyse_matrix.keys())))
lables=[]
avg_normal_index={}
print("Average Sensitivity")
for _trigger in train_analyse_matrix.keys():
    normal_si=train_analyse_matrix[_trigger]["normal_si"].tolist()[1:]
    average_sensitivity =np.array(normal_si).mean()
    avg_normal_index[_trigger]=average_sensitivity
    print(average_sensitivity)
cut_off_index=np.array(list(avg_normal_index.values())).mean()
cut_off_dict={_trigger:cut_off_index for _trigger in train_analyse_matrix.keys()}
xvalue_keys=list(train_analyse_matrix.keys())

width = .35 # width of a bar

m1_t = pd.DataFrame({
 'avg_normal_si' : list(avg_normal_index.values()),
 'cutoff_si' : list(cut_off_dict.values()),
 })

ax=m1_t['cutoff_si'].plot(secondary_y=True,color="red") 
m1_t[['avg_normal_si']].plot(kind='bar', width = width, ax=ax)
#plt.xlim([-width, len(m1_t['normal'])-width])
ax.set_xticklabels(xvalue_keys)

plt.savefig('normal_cutoff_avg_sensitivity_index.png')
plt.close()

cor_et=pd.DataFrame(avg_normal_index,index=[out_comp])
cor_et=cor_et.T
cm=plt.cm.viridis
cor_et_heat_map=sn.heatmap(cor_et,cmap=cm,linewidths=0.1,linecolor='white',annot=True)
plt.title("Figure: Corelation Matrix With Event Tracker")  
plt.savefig('correlation_matrix_with_event_tracker.png')
plt.close()

train_X=data_input_dataframe
train_Y=data_output_dataframe
test_date_data=train_event_data["Dates"].values

linear_reg(train_X, train_Y, validate_input_matrix, validate_output_matrix, test_date_data, out_company)

#print(train_Y)
svm_svr(train_X, train_Y,test_date_data,out_company)

#validate_output_matrix
arima_prediction(validate_output_matrix, test_date_data, out_company)



