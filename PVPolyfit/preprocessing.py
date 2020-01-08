from numpy import linalg, zeros, ones, hstack, asarray, vstack, array, mean, std
import itertools
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
import numpy as np
import scipy
import matplotlib.dates as mdates
from sklearn.metrics import mean_squared_error
from math import sqrt
import pvlib
from pvlib import clearsky, atmosphere, solarposition
from pvlib.location import Location
from pvlib.iotools import read_tmy3
from pvlib.irradiance import get_extra_radiation
import warnings
import time
warnings.filterwarnings("ignore")

def classify_weather_day_GM_Tina(df, clearsky_ghi_tag, meas_ghi_tag):
    X_csm = array(df[clearsky_ghi_tag].tolist())
    X_meas = array(df[meas_ghi_tag].tolist())

    # Calculate index of serenity, k
    # a k is calculated for every index

    k = abs(X_meas - X_csm) / X_csm
    k = array(k)
    #print("final vals k: ", k, "min k: ", k.min(), "max k: ", k.max())

    #TODO:
    # 1. CHANGE FREQUENCY TO 1 min/10 min
    # 2. For every Δt (10 minutes/1 hour), divide it into 3 splits
    # 3. For each split, iterate

    # Moving average
    # MA is calculated for a larger interval, i assume
    Nm = 3 # 5
    MA = []
    for i in range(len(k)):

        sumk = 0
        if i < Nm:
            for j in range(i+1):
                sumk += k[j]

        else:
            for iter in range(Nm):
                sumk += k[i - iter]

        MA.append(sumk)

    MA = array(MA) * (1 / Nm)

    #print("CHECK: make sure lengths are equal (k and MA): ", len(k), len(MA))

    # Moving function
    MF = []
    for i in range(len(k)):
        sumMF = 0
        if i < Nm:
            for j in range(i+1):
                sumMF += abs(MA[i] - k[i-iter])

        else:
            for iter in range(Nm):
                # MA does not iter
                sumMF += abs(MA[i] - k[i-iter])

        MF.append(sumMF)

    MF = array(MF)

    # Classification logic
    classification = []
    # input k and MF
    for i in range(len(k)):
        if(MF[i] > 0.05):
            # Variable
            classification.append(1)
        # k[i] = 0.4
        elif (k[i] > 0.7):
            # Cloudy
            classification.append(2)
        elif (k[i] > 0.2 or MF[i] > 0.02):
            # Slightly Cloudy
            classification.append(3)
        else:
            # Clear
            classification.append(4)

    return classification, k, MF


def data_preprocessing(df, xs, Y_tag, I_tag, cs_tag, Y_high_filter, print_info, include_preprocess):

    # data processing
    df.dropna(inplace = True)

    #df = df[df[xs[0]] > 20]
    # drop where ghi_clearsky is equal to 0 because we will be dividing by that later

    df = df[df[Y_tag] > 0]

    df = df[df[Y_tag] < Y_high_filter]

    # irradiance and temperature sensor verification
    # find all points outside of 3 sigma of Isc / Irradiance
    # replacing Isc with DC Current because no IV trace at inverter level

    if include_preprocess:
        if len(cs_tag) != 0:
            df = df[df[cs_tag] != 0]
    #if True:
        # OUTLIER REMOVAL
        old_num_rows = len(df.index)
        I_vs_Irr = array(df[I_tag].tolist()) / array(df[xs[0]].tolist())
        avg = mean(I_vs_Irr, axis = 0)
        sd = std(I_vs_Irr, axis = 0)
        sigmas = 3
        outliers = [(True if ((x < avg - sigmas * sd) or (x > avg + sigmas * sd)) else False) for x in I_vs_Irr]

        df['outlier_bool'] = outliers
        df = df[df['outlier_bool'] == False]
        df.drop(columns = ['outlier_bool'])

        if print_info:
            new_num_rows = len(df.index)
            print("Dropped {} of {} rows with I/Irr filter.".format((old_num_rows - new_num_rows), old_num_rows))

    return df

def add_ghi_to_df(df, start, end, freq, dropped_days, xs, ghi_tag, cs_tag, type_ = None):
    if type_ == 'NIST':
        # multiply GHI by 10^3 because it is in milli
        if len(ghi_tag) != 0:
            df[ghi_tag] = df[ghi_tag].apply(lambda x: x * 10**2)

    #TODO: Classify and group days/or/hours (check regression_notes.txt on Desktop)
    if type_ == '8157UCF' or 'PVLifetime' or 'VCAD':
        cocoa = Location(28.387566, -80.755984, tz='US/Eastern', altitude = 10.97)
    elif type_ == 'NIST':
        cocoa = Location(39.994425, -105.265645, tz='US/Mountain', altitude = 1623.974)
    else:
        print('No valid input.')
    times = pd.DatetimeIndex(start=datetime.strptime(start, '%m/%d/%Y %I:%M:%S %p').strftime('%Y-%m-%d'),
                            end=datetime.strptime(end, '%m/%d/%Y %I:%M:%S %p').strftime('%Y-%m-%d'),
                            freq=freq+'in', tz=cocoa.tz)
    cs = cocoa.get_clearsky(times)

    df = df[df[xs[0]] > 20]
    cs = cs[cs['ghi'] > 20]
    cs = cs.iloc[1:]
    cs_ghi = pd.DataFrame()

    if len(cs_tag) != 0:
        cs_ghi[cs_tag] = cs['ghi']

    cs_ghi.index = pd.to_datetime(cs.index, format = '%m/%d/%Y %I:%M:%S %p').strftime('%m/%d/%Y %I:%M:%S %p')

    cs = pd.merge(df, cs_ghi, how='inner', left_index = True, right_index = True)
    df = cs

    print(df.index)

    if len(df.index)!= 0:
        if str(type(df.index[0])) != "<class 'str'>":
            df_index = [i.strftime('%m/%d/%Y %I:%M:%S %p') for i in df.index]
            df.index = df_index

    return df
