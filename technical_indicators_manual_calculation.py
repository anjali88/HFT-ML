import numpy as np
import pandas as pd


# Commodity Channel Index
def cci(data, ndays):
    tp = (data['High'] + data['Low'] + data['Close']) / 3
    cci_calc = (tp - tp.rolling(ndays).mean()) / (0.015 * tp.rolling(ndays).std())
    cci_calc[np.isnan(cci_calc)] = 0
    return cci_calc


# Ease of Movement
def evm(high, low, volume, ndays):
    high = pd.Series(high)
    low = pd.Series(low)
    volume = pd.Series(volume)
    dm = ((high + low)/2) - (high.shift(1) + low.shift(1))/2
    br = (volume / 100000000) / (high - low)
    evm_calc = dm / br
    evm_ma = evm_calc.rolling(ndays).mean()
    evm_ma[np.isnan(evm_ma)] = 0
    return evm_ma


# Simple Moving Average
def sma(data, ndays):
    sma_calc = data['Close'].rolling(ndays).mean()
    sma_calc[np.isnan(sma_calc)] = 0
    return sma_calc


# Exponentially-weighted Moving Average
def ewma(data, ndays):
    ewma_calc = data['Close'].ewm(span=ndays, min_periods=ndays-1).mean()
    ewma_calc[np.isnan(ewma_calc)] = 0
    return ewma_calc


# Rate of Change (ROC)
def roc(data, ndays):
    n = data['Close'].diff(ndays)
    d = data['Close'].shift(ndays)
    roc_calc = n/d
    roc_calc[np.isnan(roc_calc)] = 0
    return roc_calc


# Compute the Bollinger Bands
def bbands(data, ndays):
    ma = data['Close'].rolling(ndays).mean()
    sd = data['Close'].rolling(ndays).std()
    ma[np.isnan(ma)] = 0
    sd[np.isnan(sd)] = 0
    data['UpperBB'] = ma + (2 * sd)
    data['LowerBB'] = ma - (2 * sd)
    return data


# Force Index
def force_index(close, volume, ndays):
    fi = pd.Series(close).diff(ndays) * volume
    fi[np.isnan(fi)] = 0
    return fi
