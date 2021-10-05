# Load library
import os 
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity,cosine_distances
from scipy import signal



# Task: amplitude Normalization
# min-max normalization 
min_max_scaler = MinMaxScaler()
# fyvals 

# minmax normalization for fxvals(ECG Voltage)
yvals_MinMax = min_max_scaler.fit_transform(yvals)

# examples
yvals_MinMax[370], yvals_MinMax[662], yvals_MinMax[947]


# Task: 1. R-R interval 나누기 + 3. R-R time frame(data points) Normalization 
# rr-interval
def minmax_norm_amplitude():
    df_rr_intervals = pd.DataFrame(columns=['RR_index', 'RR_interval', 'Norm_amplitude_yvalue'])

    for i in range(len(rpeaks)):
        if i + 1 == len(rpeaks):
            break
        else:
            rr_interval = rpeaks.iloc[i+1] - rpeaks.iloc[i]
            new_yval = yvals_MinMax[int(rpeaks.iloc[i]):int(rpeaks.iloc[i+1])]
            new_yval_1d = sum(new_yval.tolist(), [])

            # add data into dictionary(dataframe )
            tmp = pd.DataFrame({'RR_index':i, 'RR_interval':rr_interval, 'Norm_amplitude_yvalue': [new_yval_1d]})
                                # "rr_index: i " means 0:rpeak[1]-rpeak[0]
            df_rr_intervals = pd.concat([df_rr_intervals, tmp])
    return df_rr_intervals

# plt.hist(rr_intervals)# np.mean(rr_intervals), np.median(rr_intervals), np.max(rr_intervals)


## 
# resampling
from scipy import signal

p_new_yval_1ds = []
point_norm_sample = int(np.median(df_rr_intervals['RR_interval']))
#new_column = 
for i in range(len(df_rr_intervals)):
    p_new_yval = signal.resample(df_rr_intervals['Norm_amplitude_yvalue'].iloc[i], point_norm_sample)
    p_new_yval_1d = p_new_yval.tolist()
    p_new_yval_1ds.append(p_new_yval_1d)
    
#norm_df = pd.concat(df_rr_intervals, df_rr_intervals[''])
norm_df = df_rr_intervals.assign(Point_norm_sample = p_new_yval_1ds)

