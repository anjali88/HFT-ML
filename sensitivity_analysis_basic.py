import pandas as pd
import numpy as np
import math
import xlsxwriter as xl

event_threshold = 0
trigger_threshold = 0

event_data_raw = pd.read_excel('TestData/simple_event_data.xlsx')
trigger_data_raw = pd.read_excel('TestData/simple_trigger_data.xlsx')
event_data_columns = event_data_raw.columns
trigger_data_columns = trigger_data_raw.columns
event_data_array = np.array(event_data_raw.as_matrix())
trigger_data_array = np.array(trigger_data_raw.as_matrix())
event_delta_matrix = []
trigger_delta_matrix = []
sensitivity_matrix = []
normalized_sensitivity_index_matrix = []

# '---- Event data change----'
for col in event_data_array.T:
    event_deltas = []
    for a, b in zip(col, col[1:]):
        if a == b:
            event_deltas.append("n")
        else:
            event_deltas.append("y")
    event_delta_matrix.append(event_deltas)

# '---- Trigger data change ----'
for col in trigger_data_array.T:
    trigger_deltas = []
    for a, b in zip(col, col[1:]):
        if a == b:
            trigger_deltas.append("n")
        else:
            trigger_deltas.append("y")
    trigger_delta_matrix.append(trigger_deltas)

# Event data vs Trigger data comparison
for event_row in event_delta_matrix:
    for trigger_row in trigger_delta_matrix:
        sensitivity_row = []
        # list comparison
        for i, j in zip(trigger_row, event_row):
            if i == j:
                sensitivity_row.append(1)
            else:
                sensitivity_row.append(-1)
        sensitivity_matrix.append(sensitivity_row)

sensitivity_index_matrix = np.cumsum(sensitivity_matrix, axis=1)
sensitivity_index_matrix_transpose = sensitivity_index_matrix.T
for row in sensitivity_index_matrix:
    temp_row = []
    counter = 0
    for v in row.T:
        col = sensitivity_index_matrix_transpose[counter]
        sensitivity_index = ((v - min(col)) / (max(col) - min(col)))
        if math.isnan(sensitivity_index):
            sensitivity_index = 0
        temp_row.append(sensitivity_index)
        counter += 1
    normalized_sensitivity_index_matrix.append(temp_row)

workbook = xl.Workbook('results.xlsx')
worksheet = workbook.add_worksheet()

worksheet.write_column(0, 0, trigger_data_columns)

row = 0
for col, data in enumerate(np.array(normalized_sensitivity_index_matrix).T):
    worksheet.write_column(row, col+1, data)

workbook.close()
