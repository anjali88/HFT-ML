import technical_indicators as ti
import technical_indicators_manual_calculation as tim
import numpy as np


def create_complex_matrix(high, low, close, volume, period):
    ema = ti.exponential_moving_average(close, period)
    lbb = ti.lower_bollinger_band(close, period)
    ubb = ti.upper_bollinger_band(close, period)
    mbb = ti.middle_bollinger_band(close, period)
    dpo = ti.detrended_price_oscillator(close, period)
    kub = ti.keltner_upper_band(close, high, low, period)
    klb = ti.keltner_lower_band(close, high, low, period)
    kcb = ti.keltner_center_band(close, high, low, period)
    roc = ti.rate_of_change(close, period)
    cci = ti.commodity_channel_index(close, high, low, period)
    sma = ti.simple_moving_average(close, period)
    bbbw = ti.bolling_bands_bandwidth(close, period)
    mom = ti.momentum(close, period)
    mf = ti.money_flow(close, high, low, volume)
    mfi = ti.money_flow_index(close, high, low, volume, period)
    cmo = ti.chande_momentum_oscillator(close, period)
    wpr = ti.williams_percent_r(close)
    ad = ti.aroon_down(close, period)
    au = ti.aroon_up(close, period)
    vo = ti.volume_oscillator(volume, 1, 10)
    atr = ti.average_true_range(close, period)
    hma = ti.hull_moving_average(close, period)
    ten = ti.tenkansen(close, period)
    kij = ti.kijunsen(close, period)
    dss = ti.double_smoothed_stochastic(close, period)
    ndm = ti.negative_directional_movement(high, low)
    upc = ti.upper_price_channel(close, period, 0.02)
    lpc = ti.lower_price_channel(close, period, 0.02)
    po = ti.price_oscillator(close, 1, 10)
    sd = ti.standard_deviation(close, period)
    sv = ti.standard_variance(close, period)
    pdm = ti.positive_directional_movement(high, low)
    tp = ti.typical_price(close, high, low)
    rsi = ti.relative_strength_index(close, period)
    tma = ti.triangular_moving_average(close, period)
    tema = ti.triple_exponential_moving_average(close, period)
    tr = ti.true_range(close, period)

    matrix = [ema, lbb, ubb, mbb, dpo, kub, klb, kcb, roc, cci, sma, bbbw, mom, mf, mfi, cmo, wpr, ad, au, vo, atr, hma,
              ten, kij, dss, ndm, upc, lpc, po, sd, sv, tp, rsi, tma, tema, tr, pdm]
    return matrix


def create_simple_matrix(high, low, close, volume, period):
    cci = ti.commodity_channel_index(close, high, low, period)
    sma = ti.simple_moving_average(close, period)
    ema = ti.exponential_moving_average(close, period)
    roc = ti.rate_of_change(close, period)
    lbb = ti.lower_bollinger_band(close, period)
    ubb = ti.upper_bollinger_band(close, period)
    fi = tim.force_index(close, volume, period)
    evm = tim.evm(high, low, volume, period)
    matrix = [cci, sma, ema, roc, lbb, ubb, fi, evm]
    return matrix


def create_actual_matrix(close, period, short_period, long_period):
    macd = ti.moving_average_convergence_divergence(close, short_period, long_period)[long_period-1:]
    rsi = ti.relative_strength_index(close, period)[long_period-1:]
    upc = ti.upper_price_channel(close, period, 0.02)[long_period-1:]
    lpc = ti.lower_price_channel(close, period, 0.02)[long_period-1:]
    stoch_pd = ti.stochastic_percent_d(close, period)[long_period-1:]
    stoch_pk = ti.stochastic_percent_k(close, period)[long_period-1:]

    macd[np.isnan(macd)] = 0
    rsi[np.isnan(rsi)] = 0
    upc[np.isnan(upc)] = 0
    lpc[np.isnan(lpc)] = 0
    stoch_pd[np.isnan(stoch_pd)] = 0
    stoch_pk[np.isnan(stoch_pk)] = 0

    return [macd, rsi, upc, lpc, stoch_pd, stoch_pk]
