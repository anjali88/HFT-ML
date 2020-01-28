import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sensitivity_analysis as sa
import technical_indicator_matrix as tim
import xlsxwriter as xl

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
long_period = 30
short_period = 2

# Construct matrix with technical indicators (SMA, EWMA, Bollinger Bands, CCI, ROC, FI, EVM)
# Library used for technical indicators: https://github.com/kylejusticemagnuson/pyti
technical_indicator_names = ['MACD', 'RSI', 'UPC', 'LPC', 'Stochastic Percent D', 'Stochastic Percent K', 'High', 'Low', 'Close']
technical_indicator_matrix = np.array(tim.create_actual_matrix(close, period, short_period, long_period))
technical_indicator_matrix = np.vstack([technical_indicator_matrix, high[long_period-1:]])
technical_indicator_matrix = np.vstack([technical_indicator_matrix, low[long_period-1:]])
technical_indicator_matrix = np.vstack([technical_indicator_matrix, close[long_period-1:]])

matrix_placeholders = []

event_threshold = 0.05
trigger_threshold = 0.05

# Reduce close data size
close = close[long_period-1:close.shape[0]]

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

workbook = xl.Workbook('results.xlsx')
col = 0

worksheet_1 = workbook.add_worksheet(name="Without_SA")
worksheet_1.write_row(0, 0, technical_indicator_names)
for row, data in enumerate(np.array(technical_indicator_matrix).T):
    worksheet_1.write_row(row+1, col, data)

worksheet_2 = workbook.add_worksheet(name="With_SA")
worksheet_2.write_row(0, 0, filtered_technical_indicator_names)
for row, data in enumerate(np.array(filtered_data).T):
    worksheet_2.write_row(row+1, col, data)

worksheet_3 = workbook.add_worksheet(name="Sensitivity Analysis")
worksheet_3.write_row(0, 0, technical_indicator_names)
for row, data in enumerate(np.array(sensitivity_matrix).T):
    worksheet_3.write_row(row+1, col, data)

workbook.close()