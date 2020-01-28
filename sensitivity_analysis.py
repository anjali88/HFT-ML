import numpy as np
import math


# Sensitivity analysis
def analyse_sensitivity(trigger_data, event_data, trigger_threshold, event_threshold):
    trigger_data_array = np.array(trigger_data)
    event_data_array = np.array(event_data)
    trigger_delta_matrix = __construct_trigger_delta_matrix__(trigger_data_array, trigger_threshold)
    event_delta_matrix = __construct_event_delta_matrix__(event_data_array, event_threshold)
    sensitivity_matrix = __analyse__(trigger_delta_matrix, event_delta_matrix)
    sensitivity_index_matrix = np.cumsum(sensitivity_matrix, axis=1)
    normalized_sensitivity_index_matrix = __get_normalized_sensitivity__(sensitivity_index_matrix)
    return normalized_sensitivity_index_matrix

# Sensitivity analysis
def analyse_sensitivity_multiple_input(trigger_data, event_data, trigger_threshold, event_threshold,period=1):
    if period !=1:
        event_data=event_data[::period]
        for _trigger in trigger_data.keys():
            trigger_data[_trigger]["close"]=trigger_data[_trigger]["close"][::period]
    if True: #len(trigger_data) == 1:
        _trigger_data_array=[]
        trigger_delta_matrix=[]
        event_data_array = np.array(event_data)
        event_delta_matrix = __construct_event_delta_matrix__(event_data_array, event_threshold)
        trigger_delta_matrix, trigger_data_dict = __construct_multiple_trigger_delta_matrix__(trigger_data,trigger_threshold)
        #trigger_delta_matrix=np.array(trigger_delta_matrix).transpose().tolist()
        sensitivity_matrix, trigger_data_dict = __analyse_multiple__(trigger_data_dict, event_delta_matrix)
        for _trigger in trigger_data_dict.keys():
            sensitivity_index_matrix = np.cumsum(trigger_data_dict[_trigger]["si"], axis=0)
            trigger_data_dict[_trigger]["si"]=sensitivity_index_matrix
        trigger_data_dict = __get_multiple_normalized_sensitivity__(trigger_data_dict)
        print(trigger_data_dict)
        return trigger_data_dict


# '---- Event data change----'
def __construct_event_delta_matrix__(event_data_array, event_threshold):
    event_delta_matrix = []
    # for col in event_data_array:
    event_deltas = []
    for a, b in zip(event_data_array, event_data_array[1:]):
        if (b * (1 - event_threshold)) <= a <= (b * (1 + event_threshold)):
            event_deltas.append("n")
        else:
            event_deltas.append("y")
    event_delta_matrix.append(event_deltas)
    return event_delta_matrix

# '---- Trigger data change ----'
def __construct_multiple_trigger_delta_matrix__(trigger_data_array, trigger_threshold):
    trigger_data=[trigger_data_array[_trigger]["close"] for _trigger in trigger_data_array.keys()]
    trigger_delta_matrix = []
    for _tds in trigger_data_array.keys():
        _trigger=trigger_data_array[_tds]["close"]
        trigger_deltas = []
        for c, d in zip(_trigger, _trigger[1:]):
            if (d * (1 - trigger_threshold)) <= c <= (d * (1 + trigger_threshold)):
                trigger_deltas.append("n")
            else:
                trigger_deltas.append("y")
        trigger_data_array[_tds]["sa"]=np.array(trigger_deltas)
        trigger_delta_matrix.append(trigger_deltas)
    return trigger_delta_matrix, trigger_data_array


# '---- Trigger data change ----'
def __construct_trigger_delta_matrix__(trigger_data_array, trigger_threshold):
    trigger_delta_matrix = []
    for col in trigger_data_array:
        trigger_deltas = []
        for c, d in zip(col, col[1:]):
            if (d * (1 - trigger_threshold)) <= c <= (d * (1 + trigger_threshold)):
                trigger_deltas.append("n")
            else:
                trigger_deltas.append("y")
        trigger_delta_matrix.append(trigger_deltas)
    return trigger_delta_matrix


# Event data vs Trigger data comparison
def __analyse__(trigger_delta_matrix, event_delta_matrix):
    sensitivity_matrix = []
    for event_row in event_delta_matrix:
        for trigger_row in trigger_delta_matrix:
            sensitivity_row = []
            print("event_row {} trigger_row {}".format(event_row,trigger_row))
            # list comparison
            for i, j in zip(trigger_row, event_row):
                if i == j:
                    print("i={} j={}".format(i,j))
                    sensitivity_row.append(1)
                else:
                    sensitivity_row.append(-1)
            sensitivity_matrix.append(sensitivity_row)
    return sensitivity_matrix

# Event data vs Trigger data comparison
def __analyse_multiple__(trigger_delta_matrix, event_delta_matrix):
    sensitivity_matrix = []
    trigger_delta_dict=trigger_delta_matrix
    for _trigger in trigger_delta_matrix.keys():
        trigger_row=trigger_delta_matrix[_trigger]["sa"]
        for event_row in event_delta_matrix:
            sensitivity_row = []
            # list comparison
            for i, j in zip(trigger_row, event_row):
                if i == j:
                    sensitivity_row.append(1)
                else:
                    sensitivity_row.append(-1)
        sensitivity_matrix.append(sensitivity_row)
        trigger_delta_matrix[_trigger]["si"]=np.array(sensitivity_row)
    return sensitivity_matrix, trigger_delta_matrix

def __get_multiple_normalized_sensitivity__(trigger_delta_matrix):
    for _trigger in trigger_delta_matrix.keys():
        temp_row=[]
        sensitivity_index_matrix=trigger_delta_matrix[_trigger]["si"]
        lower_bound=min(sensitivity_index_matrix)
        upper_bound=max(sensitivity_index_matrix)
        for v in sensitivity_index_matrix:
            sensitivity_index=((v-lower_bound)/(upper_bound-lower_bound))
            temp_row.append(sensitivity_index)
        trigger_delta_matrix[_trigger]["normal_si"]=temp_row
    return trigger_delta_matrix
            
def __get_normalized_sensitivity__(sensitivity_index_matrix):
    normalized_sensitivity_index_matrix = []
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
    return normalized_sensitivity_index_matrix
