from pyti.exponential_moving_average import exponential_moving_average as ema
from pyti.bollinger_bands import lower_bollinger_band as lbb
from pyti.bollinger_bands import upper_bollinger_band as ubb
from pyti.bollinger_bands import middle_bollinger_band as mbb
from pyti.detrended_price_oscillator import detrended_price_oscillator as dpo
from pyti.keltner_bands import upper_band as kub
from pyti.keltner_bands import lower_band as klb
from pyti.keltner_bands import center_band as kcb
from pyti.rate_of_change import rate_of_change as roc
from pyti.commodity_channel_index import commodity_channel_index as cci
from pyti.simple_moving_average import simple_moving_average as sma
from pyti.momentum import momentum as mom
from pyti.money_flow import money_flow as mf
from pyti.money_flow_index import money_flow_index as mfi
from pyti.chande_momentum_oscillator import chande_momentum_oscillator as cmo
from pyti.aroon import aroon_down as ad
from pyti.aroon import aroon_up as au
from pyti.average_true_range import average_true_range as atr
from pyti.hull_moving_average import hull_moving_average as hma
from pyti.ichimoku_cloud import tenkansen as ten
from pyti.ichimoku_cloud import kijunsen as kij
from pyti.directional_indicators import negative_directional_movement as ndm
from pyti.price_channels import upper_price_channel as upc
from pyti.price_channels import lower_price_channel as lpc
from pyti.price_oscillator import price_oscillator as po
from pyti.standard_deviation import standard_deviation as sd
from pyti.standard_variance import standard_variance as sv
from pyti.stochrsi import stochrsi as srsi
from pyti.relative_strength_index import relative_strength_index as rsi
from pyti.triangular_moving_average import triangular_moving_average as tma
from pyti.triple_exponential_moving_average import triple_exponential_moving_average as tema
from pyti.true_range import true_range as tr
from pyti.bollinger_bands import bandwidth as bbbw
from pyti.williams_percent_r import williams_percent_r as wpr
from pyti.double_smoothed_stochastic import double_smoothed_stochastic as dss
from pyti.directional_indicators import positive_directional_movement as pdm
from pyti.typical_price import typical_price as tp
from pyti.volume_oscillator import volume_oscillator as vo
from pyti.moving_average_convergence_divergence import moving_average_convergence_divergence as macd
from pyti.stochastic import percent_d as stoch_pd
from pyti.stochastic import percent_k as stoch_pk
from pyti.volatility import volatility as vol
import numpy as np


def exponential_moving_average(data, period):
    return ema(data, period)


def lower_bollinger_band(data, period):
    return lbb(data, period)


def upper_bollinger_band(data, period):
    return ubb(data, period)


def middle_bollinger_band(data, period):
    return mbb(data, period)


def detrended_price_oscillator(data, period):
    return dpo(data, period)


def keltner_upper_band(close, high, low, period):
    return kub(close, high, low, period)


def keltner_lower_band(close, high, low, period):
    return klb(close, high, low, period)


def keltner_center_band(close, high, low, period):
    return kcb(close, high, low, period)


def rate_of_change(data, period):
    return roc(data, period)


def commodity_channel_index(close, high, low, period):
    return cci(close, high, low, period)


def simple_moving_average(data, period):
    return sma(data, period)


def bolling_bands_bandwidth(data, period):
    return bbbw(data, period)


def momentum(data, period):
    return mom(data, period)


def money_flow(close, high, low, volume):
    return mf(close, high, low, volume)


def money_flow_index(close, high, low, volume, period):
    return mfi(close, high, low, volume, period)


def chande_momentum_oscillator(data, period):
    return cmo(data, period)


def williams_percent_r(close):
    return wpr(close)


def aroon_down(data, period):
    return ad(data, period)


def aroon_up(data, period):
    return au(data, period)


def volume_oscillator(volume, short_period, long_period):
    return vo(volume, short_period, long_period)


def average_true_range(data, period):
    return atr(data, period)


def hull_moving_average(data, period):
    return hma(data, period)


def tenkansen(data, period):
    return ten(data, period)


def kijunsen(data, period):
    return kij(data, period)


def double_smoothed_stochastic(close, period):
    return dss(close, period)


def negative_directional_movement(high, low):
    return ndm(high, low)


def upper_price_channel(data, period, upper_percent):
    return np.array(upc(data, period, upper_percent))


def lower_price_channel(data, period, lower_percent):
    return np.array(lpc(data, period, lower_percent))


def price_oscillator(data, short_period, long_period):
    return po(data, short_period, long_period)


def standard_deviation(data, period):
    return sd(data, period)


def standard_variance(data, period):
    return sv(data, period)


def positive_directional_movement(high, low):
    return pdm(high, low)


def typical_price(close, high, low):
    return tp(close, high, low)


def stochrsi(data, period):
    return srsi(data, period)


def relative_strength_index(data, period):
    return rsi(data, period)


def triangular_moving_average(data, period):
    return tma(data, period)


def triple_exponential_moving_average(data, period):
    return tema(data, period)


def true_range(data, period):
    return tr(data, period)


def moving_average_convergence_divergence(data, short_period, long_period):
    return macd(data, short_period, long_period)


def stochastic_percent_k(data, period):
    return stoch_pk(data, period)


def stochastic_percent_d(data, period):
    return stoch_pd(data, period)


def adaptive_moving_average(data, period, fast_period, slow_period):
    change = np.abs([x - y for x, y in zip(data, data[period:])])
    volatility = vol(data, period)[period:]
    efficiency_ratio = change/volatility
    smoothing_constant = (efficiency_ratio * ((2 / (fast_period + 1)) - (2 / (slow_period + 1))) + (2 / slow_period + 1))**2
    initial_value = simple_moving_average(data, period)[period]
    kama = []
    data = data[period:]
    for i, value in np.ndenumerate(data):
        length = kama.__len__()
        index = i[0]
        if length > 0:
            v = kama[index-1] + smoothing_constant[index] * (value - kama[index-1])
            kama.append(v)
        else:
            v = initial_value + smoothing_constant[index] * (value - initial_value)
            kama.append(v)

    return np.array(kama)

