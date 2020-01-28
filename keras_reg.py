import numpy as np
import pandas as pd
from keras.layers import Dense
from keras.models import Sequential
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

def linear_reg(train_X, train_Y, validate_input_matrix, validate_output_matrix, test_date_data, out_company):
    model = Sequential()
    #get number of columns in training data
    n_cols = train_X.shape[1]
    
    #add model layers
    model.add(Dense(128, activation='relu', input_shape=(n_cols,)))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error',metrics=['acc'])
    early_stopping_monitor = EarlyStopping(patience=3)
    model.fit(train_X, train_Y, validation_split=0.2, epochs=100, callbacks=[early_stopping_monitor])
    model.fit(train_X, train_Y, epochs=200, batch_size=2048)

    loss_accuracy = model.evaluate(validate_input_matrix, validate_output_matrix)
    print("-;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;")
    print(loss_accuracy)
    print("-;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;")
    plt.title("Figure: Prediction of Data ({})".format(out_company))
    #test_y_predictions = model.predict(train_X)
    #plt.plot(np.array(train_analyse_matrix[_trigger]["s_index"]))
    pred = model.predict(train_X)
    #print(pred)
    #print(train_X[:5])
    #print(train_Y[:5])
    #print("-;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;")
    plt.plot(test_date_data, pred, 'r-', label='Predicted Stock with EventTracker')
    plt.plot(test_date_data, train_Y, 'b-', label='Actual Stock Value')
    plt.legend()
    plt.savefig('data_prediction.png')
    plt.close()
