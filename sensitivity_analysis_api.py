# coding: latin-1

import pandas as pd
import numpy as np
import math
from flask import Flask
from flask import request
from flask_cors import CORS, cross_origin
import xlsxwriter as xl

app = Flask(__name__)
CORS(app)
app.config.from_object(__name__)
app.debug = True


@app.route('/')
def welcome():
    return 'Welcome!'


@app.route('/data', methods=['GET'])
def analyse_data():
    event_data_raw = pd.read_excel('event_data.xlsx').as_matrix()
    trigger_data_raw = pd.read_excel('trigger_data.xlsx').as_matrix()
    event_data_array = np.array(event_data_raw)
    trigger_data_array = np.array(trigger_data_raw)
    event_delta_matrix = calculate_event_deltas(event_data_array)
    trigger_delta_matrix = calculate_trigger_deltas(trigger_data_array)
    sensitivity_matrix = compare_event_trigger_data(event_delta_matrix, trigger_delta_matrix)
    normalized_sensitivity_index_matrix = analyse_sensitivity(sensitivity_matrix)
    wb = xl.Workbook('results.xlsx')
    ws = wb.add_worksheet()
    for col in normalized_sensitivity_index_matrix:
        for row in col:
            ws.write(col, row)
    wb.close()
    return normalized_sensitivity_index_matrix


# '---- Event data ----'
def calculate_event_deltas(event_data_array):
    matrix = []
    for col in event_data_array.T:
        event_deltas = []
        for a, b in zip(col, col[1:]):
            if a == b:
                event_deltas.append("n")
            else:
                event_deltas.append("y")
        matrix.append(event_deltas)
    return matrix


# '---- Trigger data ----'
def calculate_trigger_deltas(trigger_data_array):
    matrix = []
    for col in trigger_data_array.T:
        trigger_deltas = []
        for a, b in zip(col, col[1:]):
            if a == b:
                trigger_deltas.append("n")
            else:
                trigger_deltas.append("y")
        matrix.append(trigger_deltas)
    return matrix


# Event data vs Trigger data comparison
def compare_event_trigger_data(event_delta_matrix, trigger_delta_matrix):
    matrix = []
    for event_row in event_delta_matrix:
        for trigger_row in trigger_delta_matrix:
            sensitivity_row = []
            # list comparison
            for i, j in zip(trigger_row, event_row):
                if i == j:
                    sensitivity_row.append(1)
                else:
                    sensitivity_row.append(-1)
            matrix.append(sensitivity_row)
    return matrix


def analyse_sensitivity(sensitivity_matrix):
    matrix = []
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
        matrix.append(temp_row)
    return matrix
