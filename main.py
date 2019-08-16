from PVPolyfit.core import pvpolyfit
from PVPolyfit import preprocessing as preprocess

import sys
sys.path.append('D:\\Documents\\FSEC\\OSI_Overload\\OSI Internship\\Python Scripts\\8157UCF-master')
from OSIPI_Python import Pithon_Servers as PSe

import pandas as pd
from datetime import datetime
import numpy as np
import copy
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
sns.set(style="whitegrid")

def my_query_data(xs, Y_tag, I_tag, ghi_tag, cs_tag, start, end, test_start, test_end, freq, valid_eventframes, type_):
    # package tags for query
    all_tags = xs + [Y_tag] + [I_tag] + [ghi_tag]

    ''' TRAIN DATA '''
    if (type_ == '8157UCF' or type_ == 'NIST' or type_ == 'VCAD') and filter_with_events == True:
        # using event frame filter
        print(start, end)
        df, train_dropped_days = PSe.Summarize_Multi_PIData(all_tags, start, end, freq, return_dropped_days = True, valid_eventframes = valid_eventframes)
        df.dropna(inplace=True)
        #print("Initial len df: ", len(df))
        df = preprocess.add_ghi_to_df(df, start, end, freq, train_dropped_days, xs, ghi_tag, cs_tag, type_ = type_)
        df.to_csv('D://Downloads//train_df.csv')

    else:
        df = PSe.Summarize_Multi_PIData(all_tags, start, end, freq)
        df.dropna(inplace=True)
        df = preprocess.add_ghi_to_df(df, start, end, freq, [], xs, ghi_tag, cs_tag, type_ = type_)

    ''' TEST DATA '''
    if type_ == '8157UCF' or type_ == 'NIST' or type_ == 'VCAD':
        # using eventframe filter
        print(test_start, test_end)
        test_df, test_dropped_days = PSe.Summarize_Multi_PIData(all_tags, test_start, test_end, freq, return_dropped_days = True, valid_eventframes = valid_eventframes)
        test_df.dropna(inplace = True)
        #print("Initial len test_df: ", len(test_df))
        test_df = preprocess.add_ghi_to_df(test_df, test_start, test_end, freq, test_dropped_days, xs, ghi_tag, cs_tag, type_ = type_)
        test_df.to_csv('D://Downloads//test_df.csv')
    else:
        test_df = PSe.Summarize_Multi_PIData(all_tags, test_start, test_end, freq)
        test_df.dropna(inplace = True)
        test_df = preprocess.add_ghi_to_df(test_df, test_start, test_end, freq, [], xs, ghi_tag, cs_tag, type_ = type_)

    return df, test_df
############################################################################
type_ = '8157UCF'
freq = '10m'
if type_ == '8157UCF':
    xs = ['8157_UCF.MET1.POA_irradiance', 'AmbientAir_Av']
    I_tag = 'UCF_Inverter_1.SMA_DCcurrent.764a77bd-01f9-4e2a-ae55-fcd76979a354'
    #Y_tag = 'UCF_Inverter_1.IEC61724_SMA_String2Power.ca19132f-60b0-4b34-9e04-e4b82b4ca690'
    Y_tag = 'UCF_Inverter_1.IEC61724_SMA_String1Power.749197a3-eef5-4b04-8d64-9384c6c8e625'
    #Y_tag = '8157_UCF.UCF_Inverter_1.AC_Pwr_Avg_60s'

    ghi_tag = 'Global_Irr_Av'

    valid_eventframes = ['PI-DCVolt-flatlining', 'PI-DCVolt-not-reporting', 'PI-DCVolt-outside-range-1',
                        'Irradiance-flatlining', 'Irradiance-not-reporting', 'Irradiance-outside-range',
                        'Modtemps-flatlining', 'POA_irr-flatlining', 'POA_irr-not-reporting', 'POA_irr-outside-range']


    start = '2/03/2019 12:00:00 AM'
    end = '3/16/2019 12:00:00 AM'
    test_start = '3/16/2019 12:00:00 AM'
    test_end = '4/30/2019 12:00:00 AM'
    date_list = pd.date_range(start=test_start, end = test_end, freq='W').strftime('%m/%d/%Y 12:%M:%S %p')
    test_starts = date_list[:-1]
    test_ends = date_list[1:]
    dt_test_start_list = [datetime.strptime(j, "%m/%d/%Y %H:%M:%S %p") for j in test_starts]

if type_ == 'PVLifetime':
    xs = ['8157_UCF.MET1.POA_irradiance', '8157_UCF.UCF_Inverter_1.CB_1.S_01.M_01.Temperature'] #'8157_UCF.MET1.Avg_Module_Temps']
    I_tag = 'UCF_Inverter_1.SMA_DCcurrent.764a77bd-01f9-4e2a-ae55-fcd76979a354'

    V_tag = '8157_UCF.UCF_Inverter_1.CB_1.S_01.M_01.Pmax'
    ghi_tag = 'Global_Irr_Av'

    start = '1/03/2019 12:00:00 AM'
    end = '2/18/2019 12:00:00 AM'
    test_start = '2/18/2019 12:00:00 AM'
    test_end = '2/22/2019 12:00:00 AM'

if type_ == 'NIST':
    ### NIST DATA
    I_tag = 'Ground_InvIDCin_Avg'

    Y_tag = 'Ground_InvPDC_kW_Avg'

    xs = ['Ground_RefCell1_Wm2_Avg', 'Ground_AmbTemp_C_Avg']
    ghi_tag = 'Ground_Pyra1_mV_Avg'

    start = '8/01/2016 12:00:00 AM'
    end = '8/01/2017 12:00:00 AM'


    test_start = end

    test_end = '3/20/2018 12:00:00 AM'

    date_list = pd.date_range(start=test_start, end = test_end, freq='W').strftime('%m/%d/%Y 12:%M:%S %p')
    test_starts = date_list[:-1]
    test_ends = date_list[1:]
    dt_test_start_list = [datetime.strptime(j, "%m/%d/%Y %H:%M:%S %p") for j in test_starts]
    valid_eventframes = ['PI-NIST_Irr-slope-flatlining', 'PI-NIST_Pwr-flatlining']


if type_ == 'VCAD':
    module = 'UCF_1808_040.RDE.'
    I_tag = module + 'Impp'
    Y_tag = module + 'Pmpp'
    xs = [module + 'Irradiance', 'AmbientAir_Av', 'WindSpeed_Av']#module + 'PV_Temp']
    ghi_tag = 'Global_Irr_Av'

    #start = '8/17/2018 12:00:00 AM'
    #end = '11/17/2018 12:00:00 AM'
    start = '5/1/2019 12:00:00 AM'
    end = '6/1/2019 12:00:00 AM'
    #end = '8/24/2018 12:00:00 AM'

    test_start = end
    test_end = '6/08/2019 12:00:00 AM'
    #test_end = '8/31/2018 12:00:00 AM'
    #test_start = '3/15/2019 12:00:00 AM'
    #test_end = '4/15/2019 12:00:00 AM'
    date_list = pd.date_range(start=test_start, end = test_end, freq='W').strftime('%m/%d/%Y 12:%M:%S %p')
    test_starts = date_list[:-1]
    test_ends = date_list[1:]
    dt_test_start_list = [datetime.strptime(j, "%m/%d/%Y %H:%M:%S %p") for j in test_starts]

    valid_eventframes = ['VCAD_Pwr-not-reporting', 'VCAD_DCPwr-flatlining']



num_clusters = 6
num_clusters_list = [10]
num_iterations = 1
plot_graph = 'regression'
degrees = range(2,6)
cs_tag = 'clearsky_ghi'
show_plot = False
filter_with_events = True
filename = 'D:\\nist_summer_error_clusters\\{}_full_report_1thru6.txt'.format(type_)

############################################################################



train_df, test_df = my_query_data(xs, Y_tag, I_tag, ghi_tag, cs_tag, start, end, test_start, test_end, freq, valid_eventframes, type_)

model_output, days_rmses = pvpolyfit(train_df, test_df, Y_tag, xs, I_tag, ghi_tag, cs_tag, 8, 6, 'polynomial with log(POA)', 10000, 8, plot_graph = True, graph_type = 'regression', print_info = True)
sys.exit()
#thresh = np.array(day_validation).mean()
#stddev = np.std(day_validation)
#upper_control_limit = thresh + 1*stddev
#thresh = validation_mean
#upper_control_limit = thresh + 3*validation_std

#week_rmses = []
#for i in range(len(test_starts)):
train_df, test_df = my_query_data(xs, Y_tag, I_tag, ghi_tag, cs_tag, start, end, test_start, test_end, freq, valid_eventframes, type_)
model_output, days_rmses = pvpolyfit(train_df, test_df, Y_tag, xs, I_tag, ghi_tag, cs_tag, 8, 6, 10000, 8, plot_graph = True, graph_type = 'regression', print_info = True)
#week_rmses.append(days_rmses)

# calculate weekly average rmse
avg_RMSE = sum(days_rmses) / len(days_rmses)

# Flatten out list originally: [[Day 0 modelled_P], [Day 1 modelled_P], ...]
flattened_modelled_P = [item for sublist in modelled_P for item in sublist]
max_P = max(flattened_modelled_P)

print("Average RMSE in test dataset: {:.4f} W".format(avg_RMSE))
print("Or, equivalently {:.4f}% of max".format((avg_RMSE / max_P)*100))