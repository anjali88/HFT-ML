import numpy as np
import math

# Sensitivity analysis
def analyse_sensitivity_multiple_input(trigger_data, event_data, trigger_threshold, event_threshold,period=1):
    if period !=1:
        event_data=event_data[::period]
        for _trigger in trigger_data.keys():
            trigger_data[_trigger]=trigger_data[_trigger][::period]
    if True: #len(trigger_data) == 1:
        _trigger_data_array=[]
        trigger_delta_matrix=[]
        event_data_array = np.array(event_data["Close"])
        event_delta_matrix = __construct_event_delta_matrix__(event_data_array, event_threshold)
        event_data=event_data.assign(delta_matrix=event_delta_matrix)
        trigger_delta_matrix_dict = __construct_multiple_trigger_delta_matrix__(trigger_data,trigger_threshold)
        trigger_data_dict = __analyse_multiple__(trigger_delta_matrix_dict, event_delta_matrix)
        for _trigger in trigger_data_dict.keys():
            delta_index=np.array(trigger_data_dict[_trigger]["delta_index"])
            sensitivity_index_matrix = np.cumsum(delta_index, axis=0).tolist()
            trigger_data_dict[_trigger].assign(si=sensitivity_index_matrix)
        trigger_data_dict = __get_multiple_normalized_sensitivity__(trigger_data_dict)
        return trigger_data_dict, event_data


# '---- Event data change----'
def __construct_event_delta_matrix__(event_data_array, event_threshold):
    event_delta_matrix = []
    # for col in event_data_array:
    event_deltas = []
    event_deltas.append(None)
    for a, b in zip(event_data_array, event_data_array[1:]):
        if (b * (1 - event_threshold)) <= a <= (b * (1 + event_threshold)):
            event_deltas.append("n")
        else:
            event_deltas.append("y")
    event_delta_matrix=event_deltas
    return event_delta_matrix

# '---- Trigger data change ----'
def __construct_multiple_trigger_delta_matrix__(trigger_data_array, trigger_threshold):
    for _tds in trigger_data_array.keys():
        _trigger=trigger_data_array[_tds]["Close"].tolist()
        #_trigger=np.array(_trigger)
        trigger_deltas=[]
        trigger_deltas.append(None)
        for c, d in zip(_trigger, _trigger[1:]):
            if (d * (1 - trigger_threshold)) <= c <= (d * (1 + trigger_threshold)):
                trigger_deltas.append("n")
            else:
                trigger_deltas.append("y")
        trigger_data_array[_tds]=trigger_data_array[_tds].assign(delta_matrix=trigger_deltas)
    return trigger_data_array

# Event data vs Trigger data comparison
def __analyse_multiple__(trigger_delta_matrix, event_delta_matrix):
    sensitivity_matrix = []
    trigger_delta_dict=trigger_delta_matrix
    event_row=event_delta_matrix[1:]
    for _trigger in trigger_delta_matrix.keys():
        trigger_row=trigger_delta_matrix[_trigger]["delta_matrix"].tolist()[1:]
        sensitivity_row = []
        sensitivity_row.append(0)
        for i, j in zip(trigger_row, event_row):
            if i == j:
                sensitivity_row.append(1)
            else:
                sensitivity_row.append(-1)
        sensitivity_matrix.append(sensitivity_row)
        trigger_delta_matrix[_trigger]=trigger_delta_matrix[_trigger].assign(delta_index=sensitivity_row)
        s_index=np.cumsum(sensitivity_row, axis=0)
        trigger_delta_matrix[_trigger]=trigger_delta_matrix[_trigger].assign(s_index=s_index)
    return trigger_delta_matrix

def __get_multiple_normalized_sensitivity__(trigger_delta_matrix):
    for _trigger in trigger_delta_matrix.keys():
        temp_row=[]
        temp_row.append(None)
        sensitivity_index_matrix=trigger_delta_matrix[_trigger]["s_index"].tolist()[1:]
        lower_bound=min(sensitivity_index_matrix)
        upper_bound=max(sensitivity_index_matrix)
        for v in sensitivity_index_matrix:
            sensitivity_index=((v-lower_bound)/(upper_bound-lower_bound))
            temp_row.append(sensitivity_index)
        trigger_delta_matrix[_trigger]=trigger_delta_matrix[_trigger].assign(normal_si=temp_row)
    return trigger_delta_matrix
