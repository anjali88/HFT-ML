import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt

def svm_svr(X,y, test_date_data, out_company):
    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
    # #############################################################################
    # Fit regression model
    y_lin = svr_rbf.fit(X, y.values.ravel()).predict(X)
    #print(y_lin)
    # Look at the results
    lw=2
    plt.plot(test_date_data, y_lin, color='c', lw=lw, label='Predicted (EventTracker)')
    #plt.plot(test_date_data, y, color='b', lw=lw, label='Predicted (EventTracker)')
    plt.xlabel('date')
    plt.ylabel('target')
    plt.title('Support Vector Regression {}'.format(out_company))
    plt.legend()
    plt.savefig('data_prediction_svr.png')
    plt.close()
