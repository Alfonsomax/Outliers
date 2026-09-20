# -*- coding: utf-8 -*-
import sys
import os
import pandas as pd
import numpy as np
import time
import pathlib
from datetime import datetime
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL
import statsmodels.api as sm
import random
from scipy.stats import zscore
from scipy.stats.mstats import winsorize
import matplotlib.dates as mdates


main_path = pathlib.Path(__file__).parent.resolve()
os.chdir(main_path)

##########################################################################################
# Auxiliary Modules Main Path
##########################################################################################

aux_func_path = os.path.join(main_path, "aux_func")
if aux_func_path not in sys.path:
    sys.path.append(aux_func_path)
inputs_path = os.path.join(main_path, "inputs")

from df_creator_module import df_creator


if "db_creator" not in globals():
    from db_creator_reader_module import db_creator
if "db_reader" not in globals():
    from db_creator_reader_module import db_reader

##########################################################################################

t000 = time.time()

pd.set_option('display.max_colwidth', 300)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 200)
pd.set_option('display.width', 1000)


###############################################################################
#%% Control Inputs
###############################################################################

# Today
today = datetime.today()
# Script Starting Date
SCRIPT_EXECUTION_DATE = today.strftime("%Y%m%d")
# Input data for the day
FILE_DATE_INPUT = '20251020'
init_date = datetime.strptime(FILE_DATE_INPUT, '%Y%m%d')
init_date_folder = init_date.strftime('%Y-%m-%d')


# Adjustable parameters
RESAMPLE_RULE = None        # 'W' for weekly, 'M' for monthly, None = daily
MIN_POINTS = 7
Z_THRESHOLD = 3.0
STL_THRESHOLD = 3.0         # for residuals
STL_PERIOD = 365            # 7 for daily data with weekly seasonality; None = attempt to infer
manual_season = ['A', 'B', 'C']  # manual stationarity
USE_ROBUST = False          # Use the median/MAD for a robust z-score

# Variable to enable debug mode. When debug mode is enabled, it generates intermediate process files and warning messages.
DEBUG_MODE_ON = True



###############################################################################
# Hypotheses, internal variables, and pathways
###############################################################################

Days_month = 30
DAYS_RECEPTION_DELTA = 21
FILE_DATE_INIT = pd.to_datetime(FILE_DATE_INPUT)
Fecha_inicio_fcst = (FILE_DATE_INIT.normalize() + pd.offsets.MonthEnd(0)) + pd.Timedelta(days=1)
FILE_DATE_INPUT_INIT_PATH = f'\\{FILE_DATE_INPUT}'
path_initial = r"rute/"
path_initial_compare = r"rute/"

db_path = os.path.join(main_path, "sqlite3", "INPUT_" + FILE_DATE_INPUT + ".db")


CHECK_UPDATE = True
#CHECK_UPDATE = False

if CHECK_UPDATE == True:
    df_group = df_creator(path_initial, "latin1", "SHEETNAME")
    df_compare = df_creator(path_initial_compare, "latin1", "SHEETNAME")
else:
    df_group = db_reader(db_path, "df_group")
    df_compare = db_reader(db_path, "df_compare")


###############################################################################
#%% DATA Process
###############################################################################
def Normalize_Process(df_group):

    if DEBUG_MODE_ON:
        print("\n DEBUG ** Normalize Process")

    # Filter normalization
    df_group = df_group.dropna(how='all')
    df_group = df_group[~df_group.iloc[:, 0].astype(str).str.contains(r'rows selected', regex=True)]
    df_group[['DOCUMENT', 'POSITION']] = df_group['NOTE'].str.extract(r'([A-Za-z0-9]+)_(\d+)')
    df_group = df_group.drop(columns=['NOTE'])
    df_group['PART_ID'] = df_group['PART_ID'].dropna().astype(str).str.strip()
    df_group['DOCUMENT'] = df_group['DOCUMENT'].astype(str).str.strip()
    df_group['POSITION'] = df_group['POSITION'].dropna()
    df_group['POSITION'] = pd.to_numeric(df_group['POSITION'], errors='coerce')
    df_group['POSITION'] = df_group['POSITION'].astype(int)
    df_group['HISTORYDATE'] = pd.to_datetime(df_group['HISTORYDATE'], format='%Y%m%d')
    df_group['HISTORYDATE'] = df_group['HISTORYDATE'].dt.strftime('%Y-%m-%d')
    df_group['QTY'] = pd.to_numeric(df_group['QTY'], errors='coerce')

    if DEBUG_MODE_ON:
        print(f"number of rows df_group: {df_group.shape[0]}")
        print(f"number of columns df_group: {df_group.shape[1]}")

    return df_group


def Normalize_compare(df_compare):

    if DEBUG_MODE_ON:
        print("\n DEBUG ** Normalize_compare Process")

    # Filter normalization
    df_compare['MATERIAL'] = df_compare['MATERIAL'].astype(str).str.strip()
    df_compare['ORDER_REASON'] = df_compare['ORDER_REASON'].astype(str).str.strip()
    df_compare['DOCUMENT2'] = df_compare['DOCUMENT2'].astype(str).str.strip()
    df_compare = df_compare[~df_compare['DOCUMENT2'].isin(['nan', '', ' '])]
    df_compare = df_compare[~df_compare['POSITION'].isin(['EMPTY', '', ' '])]
    df_compare['POSITION'] = pd.to_numeric(df_compare['POSITION'], errors='coerce')
    df_compare['POSITION'] = df_compare['POSITION'].astype(int)
    df_compare = df_compare[pd.to_datetime(df_compare['DATE'], format='%Y%m%d', errors='coerce').notna()]
    df_compare['DATE'] = pd.to_datetime(df_compare['DATE'], format='%Y%m%d')
    df_compare['DATE'] = df_compare['DATE'].dt.strftime('%Y-%m-%d')
    df_compare = df_group.merge(df_compare, left_on=['DOCUMENT', 'POSITION'], right_on=['DOCUMENT2', 'POSITION'], how='inner')

    if DEBUG_MODE_ON:
        print(f"number of rows df_compare: {df_compare.shape[0]}")
        print(f"number of columns df_compare: {df_compare.shape[1]}")

    return df_compare


def z_score_process(group):

    group['HISTORYDATE'] = pd.to_datetime(group['HISTORYDATE'], format='%Y-%m-%d')
    group = group.sort_values('HISTORYDATE')
    group = group.dropna(subset=['QTY'])

    std_dev = group['QTY'].std()

    if std_dev == 0 or pd.isna(std_dev):
        group.loc[:, 'Z_outlier'] = '-'
        group.loc[:, 'Z_ADJUST'] = 0
        return group

    seasonal_manual = np.where(group['ORDER_REASON'].isin(manual_season), group['QTY'], 0)
    seasonal_manual = pd.Series(seasonal_manual, index=group.index)
    group['Z_score'] = zscore(group['QTY'] - seasonal_manual)

    group.loc[group['Z_score'] > Z_THRESHOLD, 'Z_outlier'] = 'YES (high)'
    group.loc[group['Z_score'] < -Z_THRESHOLD, 'Z_outlier'] = 'YES (low)'

    group['Z_ADJUST'] = 0

    mean_cantidad = group['QTY'].mean()
    upper_limit = mean_cantidad + Z_THRESHOLD * std_dev

    group.loc[group['Z_outlier'] == 'YES (high)', 'Z_ADJUST'] = round(upper_limit)

    mask = group['QTY'] <= group['Z_ADJUST']
    group.loc[mask, 'Z_outlier'] = ''
    group.loc[mask, 'Z_ADJUST'] = 0

    return group


def stl_process(group):

    if len(group) < 10:
        group.loc[:, 'STL_outlier'] = '-'
        group.loc[:, 'STL_ADJUST'] = 0
        return group

    group['HISTORYDATE'] = pd.to_datetime(group['HISTORYDATE'], format='%Y-%m-%d')
    group = group.set_index('HISTORYDATE').sort_index()
    group = group.dropna(subset=['QTY'])

    original_idx = group.index

    group = group.reset_index().rename(columns={'index': 'HISTORYDATE'})

    seasonal_manual = np.where(group['ORDER_REASON'].isin(manual_season), group['QTY'], 0)
    seasonal_manual = pd.Series(seasonal_manual, index=group.index)

    # Non-Robust (nr)
    stl_nr = STL(
        group['QTY'] - seasonal_manual,
        period=STL_PERIOD,
        robust=False
    )
    result_nr = stl_nr.fit()

    seasonal_nr, trend_nr, resid_nr = result_nr.seasonal, result_nr.trend, result_nr.resid

    group['STL_resid'] = resid_nr

    resid_nr_mean = resid_nr.mean()
    resid_nr_std = resid_nr.std()
    #lower_bound = resid_nr_mean - 3*resid_nr_std
    #upper_bound = resid_nr_mean + 3*resid_nr_std
    lower_bound = np.percentile(resid_nr, 5.0)
    upper_bound = np.percentile(resid_nr, 95.0)

    mask_manual = group['ORDER_REASON'].isin(manual_season)
    mask_low = (group['STL_resid'] < lower_bound)
    mask_high = (group['STL_resid'] > upper_bound)

    mask_low = mask_low & (~mask_manual)
    mask_high = mask_high & (~mask_manual)

    group.loc[mask_low, 'STL_outlier'] = 'YES (low)'
    group.loc[mask_high, 'STL_outlier'] = 'YES (high)'

    adjusted_value = trend_nr + seasonal_nr + upper_bound
    group['STL_ADJUST'] = 0

    idx_high = group.index[group['STL_outlier'] == 'YES (high)']

    #upper_percentile = np.percentile(group['QTY'], 95.0)
    #group.loc[idx_high, 'STL_ADJUST'] = round(upper_percentile)
    group.loc[idx_high, 'STL_ADJUST'] = (trend_nr[idx_high] + seasonal_nr[idx_high] + upper_bound).round()

    mask_both_outliers = ((group['Z_ADJUST'] > 0) & (group['STL_ADJUST'] > 0))
    mask_some_outliers = ((group['Z_ADJUST'] > 0) | (group['STL_ADJUST'] > 0))
    upper_percentile = np.percentile(group['QTY'], 95.0)
    group.loc[mask_some_outliers, 'Winsorization'] = round(upper_percentile)
    relative_diff = (abs(group['Z_ADJUST'] - group['Winsorization'])) / group['QTY']
    diff_p90 = np.percentile(relative_diff, 90.0)
    mask_discrepancy = (
        (relative_diff > diff_p90) &
        (group['Z_ADJUST'] > 0) &
        (group['Winsorization'] > 0)
    )

    mask_winz = (group['QTY'] <= group['Winsorization'] & ~group['Z_outlier'].astype(str).str.contains('YES', na=False))
    group.loc[mask_winz, 'STL_outlier'] = ''
    group.loc[mask_winz, 'STL_ADJUST'] = 0
    group.loc[mask_winz, 'Winsorization'] = 0

    group.loc[mask_both_outliers, 'Min-adjust'] = np.where(
        # distance from the Z adjustment to the original value
        (group.loc[mask_both_outliers, 'Z_ADJUST'] -
        group.loc[mask_both_outliers, 'QTY']).abs() <=
        # distance from the STL adjustment to the original value
        (group.loc[mask_both_outliers, 'Winsorization'] -
        group.loc[mask_both_outliers, 'QTY']).abs(),
        # if the Z distance is less than or equal, use Z_ADJUST
        group.loc[mask_both_outliers, 'Z_ADJUST'],
        # otherwise use Winsorization
        group.loc[mask_both_outliers, 'Winsorization']
    )

    group.loc[mask_both_outliers, 'Max-capping'] = np.minimum(group.loc[mask_both_outliers, 'Z_ADJUST'], group.loc[mask_both_outliers, 'Winsorization'])

    group.loc[mask_both_outliers, 'Med'] = round((group.loc[mask_both_outliers, 'Z_ADJUST'] + group.loc[mask_both_outliers, 'Winsorization']) / 2)

    group.loc[mask_discrepancy, 'LOG'] = round(np.exp(
        (np.log(group.loc[mask_discrepancy, 'Z_ADJUST']) +
        np.log(group.loc[mask_discrepancy, 'Winsorization'])) / 2
    ))

    return group


"""def robust_z_scores(x):

    med = np.median(x)
    mad = np.median(np.abs(x - med))

    if mad == 0:
        std = np.std(x)
        if std == 0:
            return np.zeros_like(x)
        return (x - med) / std
    return 0.6756 * (x - med) / mad"""




###############################################################################
#%% Main Body
###############################################################################

# Activate filter functions
df_group = Normalize_Process(df_group)
df_compare = Normalize_compare(df_compare)
df_compare = df_compare.groupby(['PART_ID', 'HISTORYDATE'], as_index=False).agg({
    'QTY': 'sum',
    'DOCUMENT': 'first',
    'TIPO_DOCUMENTO': 'first',
    'ORDER_REASON': 'first'
})

df_sorted = df_compare.sort_values(
    by=['PART_ID', 'HISTORYDATE']
)


df_sorted['Z_score'] = 0.0
df_sorted['Z_outlier'] = ''
df_sorted['STL_resid'] = 0.0
df_sorted['STL_outlier'] = ''
df_sorted['Z_ADJUST'] = 0
df_sorted['STL_ADJUST'] = 0
df_sorted['Winsorization'] = 0
df_sorted['-'] = '-'
df_sorted['Min-adjust'] = 0
df_sorted['Max-capping'] = 0
df_sorted['Med'] = 0
df_sorted['LOG'] = 0


if DEBUG_MODE_ON:
        print("\n DEBUG ** Z-score analysis")
df_sorted = df_sorted.groupby('PART_ID').apply(z_score_process).reset_index(drop=True)

if DEBUG_MODE_ON:
        print("\n DEBUG ** STL analysis")
df_results = df_sorted.groupby('PART_ID', group_keys=False).apply(stl_process).reset_index(drop=True)


if DEBUG_MODE_ON:

    # ------------------------------------------------------------
    # 1. Common flag: the row is an outlier if any method detects it
    # ------------------------------------------------------------
    df_results['ANY_OUTLIER'] = (
        (df_results['Z_outlier'].astype(str).str.contains('YES', na=False)) |
        (df_results['STL_outlier'].astype(str).str.contains('YES', na=False))
    )

    # ------------------------------------------------------------
    # 2. Size of each group (number of observations)
    # ------------------------------------------------------------
    group_sizes = (
        df_results
        .groupby('PART_ID')
        .size()
        .reset_index(name='n_obs')
    )

    # ------------------------------------------------------------
    # 3. Indicate whether the group has at least one outlier
    # ------------------------------------------------------------
    group_has_outlier = (
        df_results
        .groupby('PART_ID')['ANY_OUTLIER']
        .sum()
        .reset_index(name='has_outlier')
    )

    # ------------------------------------------------------------
    # 4. Merge group size + outlier info
    # ------------------------------------------------------------
    group_summary = (
        group_sizes
        .merge(group_has_outlier, on='PART_ID', how='left')
        .sort_values(by='n_obs', ascending=False)
    )

    group_summary_2plus = group_summary[group_summary['n_obs'] >= 4]

    # ------------------------------------------------------------
    # 5. PRINT: largest groups with outlier indicator
    # ------------------------------------------------------------
    print("\nMaterials with the highest number of observations:\n")
    print(group_summary_2plus.head(15))




if DEBUG_MODE_ON:
        print("\n DEBUG ** Graphics")
########################################
# Charts
########################################


###################
# Graph1
###################
df_graph1 = df_results[df_results['PART_ID'] == '46-2490-1']
df_graph1['HISTORYDATE'] = pd.to_datetime(df_graph1['HISTORYDATE'], format='%Y-%m-%d')
df_graph1 = df_graph1.set_index('HISTORYDATE').sort_index()

full_idx = pd.date_range(start=df_graph1.index.min(), end=df_graph1.index.max(), freq='D')
df_graph1 = df_graph1.reindex(full_idx, fill_value=0)
df_graph1 = df_graph1.reset_index().rename(columns={'index': 'HISTORYDATE'})
x_dates = df_graph1['HISTORYDATE']

##########
# Z-score
##########
plt.figure(figsize=(10, 6))

mask_graph1_z = df_graph1['Z_outlier'] == 'YES (high)'
plt.scatter(x_dates[~mask_graph1_z], df_graph1['Z_score'][~mask_graph1_z], label='Data Points')
plt.scatter(df_graph1['HISTORYDATE'][mask_graph1_z], df_graph1['Z_score'][mask_graph1_z], color='red', label='Outlier')

plt.xlabel('Date')
plt.ylabel('Z_score')
plt.title('Date vs Z_score')
plt.legend()
plt.grid(True)

plot1_path = os.path.join(main_path, 'output', 'Graph1_ZScore.png')
plt.savefig(plot1_path)


######
# STL
######

seasonal_manual = np.where(df_graph1['ORDER_REASON'].isin(manual_season), df_graph1['QTY'], 0)
seasonal_manual = pd.Series(seasonal_manual, index=df_graph1.index)

# Non-robust
stl_graph1 = STL(df_graph1['QTY'] - seasonal_manual, period=STL_PERIOD, robust=False)
result_non_robust = stl_graph1.fit()

# Robust
stl_graph1_robust = STL(df_graph1['QTY'] - seasonal_manual, period=STL_PERIOD, robust=True)
result_robust = stl_graph1_robust.fit()


fig, axs = plt.subplots(4, 1, figsize=(12, 10), sharex=True)


# Raw
axs[0].plot(x_dates, df_graph1['QTY'], color='C0')
axs[0].set_ylabel('Quantity')
axs[0].set_title(f'PN - {df_graph1["PART_ID"].iloc[0]}')

# Trend
axs[1].plot(x_dates, result_non_robust.trend, label='Non-Robust', color='C0')
axs[1].plot(x_dates, result_robust.trend, '--', label='Robust', color='C1')
axs[1].legend(frameon=False)
axs[1].set_ylabel('Trend')

# Seasonal
axs[2].plot(x_dates, result_non_robust.seasonal + seasonal_manual, label='Non-Robust', color='C0')
axs[2].plot(x_dates, result_robust.seasonal + seasonal_manual, '--', label='Robust', color='C1')
axs[2].legend(frameon=False)
axs[2].set_ylabel('Seasonal')

# Residual
mask_graph1_stl = df_graph1['STL_outlier'] == 'YES (high)'
axs[3].plot(x_dates[~mask_graph1_stl], result_non_robust.resid[~mask_graph1_stl], 'o', label='Non-Robust', color='C0')
axs[3].plot(x_dates[~mask_graph1_stl], result_robust.resid[~mask_graph1_stl], 'x', label='Robust', color='C1')
axs[3].plot(x_dates[mask_graph1_stl], result_non_robust.resid[mask_graph1_stl], 'o', label='Outlier-nr', color='red')
axs[3].plot(x_dates[mask_graph1_stl], result_robust.resid[mask_graph1_stl], 'x', label='Outlier-r', color='red')
axs[3].legend(frameon=False)
axs[3].set_ylabel('Residual')
axs[3].set_xlabel('Date')


locator = mdates.MonthLocator(interval=3)
formatter = mdates.DateFormatter('%Y %b')
for ax in axs:
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)

fig.autofmt_xdate()

plt.tight_layout()
plt.savefig(os.path.join(main_path, 'output', 'Graph1_STL.png'))



###################
# Graph2
###################
df_graph2 = df_results[df_results['PART_ID'] == 'ABS0370-01']
df_graph2['HISTORYDATE'] = pd.to_datetime(df_graph2['HISTORYDATE'], format='%Y-%m-%d')
df_graph2 = df_graph2.set_index('HISTORYDATE').sort_index()

full_idx = pd.date_range(start=df_graph2.index.min(), end=df_graph2.index.max(), freq='D')
df_graph2 = df_graph2.reindex(full_idx, fill_value=0)
df_graph2 = df_graph2.reset_index().rename(columns={'index': 'HISTORYDATE'})
y_dates = df_graph2['HISTORYDATE']

##########
# Z-score
##########
plt.figure(figsize=(10, 6))

mask_graph2_z = df_graph2['Z_outlier'] == 'YES (high)'
plt.scatter(y_dates[~mask_graph2_z], df_graph2['Z_score'][~mask_graph2_z], label='Data Points')
plt.scatter(df_graph2['HISTORYDATE'][mask_graph2_z], df_graph2['Z_score'][mask_graph2_z], color='red', label='Outlier')

plt.xlabel('Date')
plt.ylabel('Z_score')
plt.title('Date vs Z_score')
plt.legend()
plt.grid(True)

plot2_path = os.path.join(main_path, 'output', 'Graph2_ZScore.png')
plt.savefig(plot2_path)


######
# STL
######

seasonal_manual = np.where(df_graph2['ORDER_REASON'].isin(manual_season), df_graph2['QTY'], 0)
seasonal_manual = pd.Series(seasonal_manual, index=df_graph2.index)

# Non-robust
stl_graph2 = STL(df_graph2['QTY'] - seasonal_manual, period=STL_PERIOD, robust=False)
result_non_robust = stl_graph2.fit()

# Robust
stl_graph2_robust = STL(df_graph2['QTY'] - seasonal_manual, period=STL_PERIOD, robust=True)
result_robust = stl_graph2_robust.fit()


fig, axs = plt.subplots(4, 1, figsize=(12, 10), sharex=True)


# Raw
axs[0].plot(y_dates, df_graph2['QTY'], color='C0')
axs[0].set_ylabel('Quantity')
axs[0].set_title(f'PN - {df_graph2["PART_ID"].iloc[0]}')

# Trend
axs[1].plot(y_dates, result_non_robust.trend, label='Non-Robust', color='C0')
axs[1].plot(y_dates, result_robust.trend, '--', label='Robust', color='C1')
axs[1].legend(frameon=False)
axs[1].set_ylabel('Trend')

# Seasonal
axs[2].plot(y_dates, result_non_robust.seasonal + seasonal_manual, label='Non-Robust', color='C0')
axs[2].plot(y_dates, result_robust.seasonal + seasonal_manual, '--', label='Robust', color='C1')
axs[2].legend(frameon=False)
axs[2].set_ylabel('Seasonal')

# Residual
mask_graph2_stl = df_graph2['STL_outlier'] == 'YES (high)'
axs[3].plot(y_dates[~mask_graph2_stl], result_non_robust.resid[~mask_graph2_stl], 'o', label='Non-Robust', color='C0')
axs[3].plot(y_dates[~mask_graph2_stl], result_robust.resid[~mask_graph2_stl], 'x', label='Robust', color='C1')
axs[3].plot(y_dates[mask_graph2_stl], result_non_robust.resid[mask_graph2_stl], 'o', label='Outlier-nr', color='red')
axs[3].plot(y_dates[mask_graph2_stl], result_robust.resid[mask_graph2_stl], 'x', label='Outlier-r', color='red')
axs[3].legend(frameon=False)
axs[3].set_ylabel('Residual')
axs[3].set_xlabel('Date')


locator = mdates.MonthLocator(interval=3)
formatter = mdates.DateFormatter('%Y %b')
for ax in axs:
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)

fig.autofmt_xdate()

plt.tight_layout()
plt.savefig(os.path.join(main_path, 'output', 'Graph2_STL.png'))



###################
# Graph3
###################
df_graph3 = df_results[df_results['PART_ID'] == 'MS24665-153']
df_graph3['HISTORYDATE'] = pd.to_datetime(df_graph3['HISTORYDATE'], format='%Y-%m-%d')
df_graph3 = df_graph3.set_index('HISTORYDATE').sort_index()

full_idx = pd.date_range(start=df_graph3.index.min(), end=df_graph3.index.max(), freq='D')
df_graph3 = df_graph3.reindex(full_idx, fill_value=0)
df_graph3 = df_graph3.reset_index().rename(columns={'index': 'HISTORYDATE'})
x_dates = df_graph3['HISTORYDATE']

##########
# Z-score
##########
plt.figure(figsize=(10, 6))

mask_graph1_z = df_graph3['Z_outlier'] == 'YES (high)'
plt.scatter(x_dates[~mask_graph1_z], df_graph3['Z_score'][~mask_graph1_z], label='Data Points')
plt.scatter(df_graph3['HISTORYDATE'][mask_graph1_z], df_graph3['Z_score'][mask_graph1_z], color='red', label='Outlier')

plt.xlabel('Date')
plt.ylabel('Z_score')
plt.title('Date vs Z_score')
plt.legend()
plt.grid(True)

plot1_path = os.path.join(main_path, 'output', 'Graph3_ZScore.png')
plt.savefig(plot1_path)


######
# STL
######

seasonal_manual = np.where(df_graph3['ORDER_REASON'].isin(manual_season), df_graph3['QTY'], 0)
seasonal_manual = pd.Series(seasonal_manual, index=df_graph3.index)

# Non-robust
stl_graph1 = STL(df_graph3['QTY'] - seasonal_manual, period=STL_PERIOD, robust=False)
result_non_robust = stl_graph1.fit()

# Robust
stl_graph1_robust = STL(df_graph3['QTY'] - seasonal_manual, period=STL_PERIOD, robust=True)
result_robust = stl_graph1_robust.fit()


fig, axs = plt.subplots(4, 1, figsize=(12, 10), sharex=True)


# Raw
axs[0].plot(x_dates, df_graph3['QTY'], color='C0')
axs[0].set_ylabel('Quantity')
axs[0].set_title(f'PN - {df_graph3["PART_ID"].iloc[0]}')

# Trend
axs[1].plot(x_dates, result_non_robust.trend, label='Non-Robust', color='C0')
axs[1].plot(x_dates, result_robust.trend, '--', label='Robust', color='C1')
axs[1].legend(frameon=False)
axs[1].set_ylabel('Trend')

# Seasonal
axs[2].plot(x_dates, result_non_robust.seasonal + seasonal_manual, label='Non-Robust', color='C0')
axs[2].plot(x_dates, result_robust.seasonal + seasonal_manual, '--', label='Robust', color='C1')
axs[2].legend(frameon=False)
axs[2].set_ylabel('Seasonal')

# Residual
mask_graph1_stl = df_graph3['STL_outlier'] == 'YES (high)'
axs[3].plot(x_dates[~mask_graph1_stl], result_non_robust.resid[~mask_graph1_stl], 'o', label='Non-Robust', color='C0')
axs[3].plot(x_dates[~mask_graph1_stl], result_robust.resid[~mask_graph1_stl], 'x', label='Robust', color='C1')
axs[3].plot(x_dates[mask_graph1_stl], result_non_robust.resid[mask_graph1_stl], 'o', label='Outlier-nr', color='red')
axs[3].plot(x_dates[mask_graph1_stl], result_robust.resid[mask_graph1_stl], 'x', label='Outlier-r', color='red')
axs[3].legend(frameon=False)
axs[3].set_ylabel('Residual')
axs[3].set_xlabel('Date')


locator = mdates.MonthLocator(interval=3)
formatter = mdates.DateFormatter('%Y %b')
for ax in axs:
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)

fig.autofmt_xdate()

plt.tight_layout()
plt.savefig(os.path.join(main_path, 'output', 'Graph3_STL.png'))


###################
# Graph4
###################
df_graph4 = df_results[df_results['PART_ID'] == 'CAN66028A']
df_graph4['HISTORYDATE'] = pd.to_datetime(df_graph4['HISTORYDATE'], format='%Y-%m-%d')
df_graph4 = df_graph4.set_index('HISTORYDATE').sort_index()

full_idx = pd.date_range(start=df_graph4.index.min(), end=df_graph4.index.max(), freq='D')
df_graph4 = df_graph4.reindex(full_idx, fill_value=0)
df_graph4 = df_graph4.reset_index().rename(columns={'index': 'HISTORYDATE'})
x_dates = df_graph4['HISTORYDATE']

##########
# Z-score
##########
plt.figure(figsize=(10, 6))

mask_graph1_z = df_graph4['Z_outlier'] == 'YES (high)'
plt.scatter(x_dates[~mask_graph1_z], df_graph4['Z_score'][~mask_graph1_z], label='Data Points')
plt.scatter(df_graph4['HISTORYDATE'][mask_graph1_z], df_graph4['Z_score'][mask_graph1_z], color='red', label='Outlier')

plt.xlabel('Date')
plt.ylabel('Z_score')
plt.title('Date vs Z_score')
plt.legend()
plt.grid(True)

plot1_path = os.path.join(main_path, 'output', 'Graph4_ZScore.png')
plt.savefig(plot1_path)


######
# STL
######

seasonal_manual = np.where(df_graph4['ORDER_REASON'].isin(manual_season), df_graph4['QTY'], 0)
seasonal_manual = pd.Series(seasonal_manual, index=df_graph4.index)

# Non-robust
stl_graph1 = STL(df_graph4['QTY'] - seasonal_manual, period=STL_PERIOD, robust=False)
result_non_robust = stl_graph1.fit()

# Robust
stl_graph1_robust = STL(df_graph4['QTY'] - seasonal_manual, period=STL_PERIOD, robust=True)
result_robust = stl_graph1_robust.fit()

fig, axs = plt.subplots(4, 1, figsize=(12, 10), sharex=True)


# Raw
axs[0].plot(x_dates, df_graph4['QTY'], color='C0')
axs[0].set_ylabel('Quantity')
axs[0].set_title(f'PN - {df_graph4["PART_ID"].iloc[0]}')

# Trend
axs[1].plot(x_dates, result_non_robust.trend, label='Non-Robust', color='C0')
axs[1].plot(x_dates, result_robust.trend, '--', label='Robust', color='C1')
axs[1].legend(frameon=False)
axs[1].set_ylabel('Trend')

# Seasonal
axs[2].plot(x_dates, result_non_robust.seasonal + seasonal_manual, label='Non-Robust', color='C0')
axs[2].plot(x_dates, result_robust.seasonal + seasonal_manual, '--', label='Robust', color='C1')
axs[2].legend(frameon=False)
axs[2].set_ylabel('Seasonal')

# Residual
mask_graph1_stl = df_graph4['STL_outlier'] == 'YES (high)'
axs[3].plot(x_dates[~mask_graph1_stl], result_non_robust.resid[~mask_graph1_stl], 'o', label='Non-Robust', color='C0')
axs[3].plot(x_dates[~mask_graph1_stl], result_robust.resid[~mask_graph1_stl], 'x', label='Robust', color='C1')
axs[3].plot(x_dates[mask_graph1_stl], result_non_robust.resid[mask_graph1_stl], 'o', label='Outlier-nr', color='red')
axs[3].plot(x_dates[mask_graph1_stl], result_robust.resid[mask_graph1_stl], 'x', label='Outlier-r', color='red')
axs[3].legend(frameon=False)
axs[3].set_ylabel('Residual')
axs[3].set_xlabel('Date')


locator = mdates.MonthLocator(interval=3)
formatter = mdates.DateFormatter('%Y %b')
for ax in axs:
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)

fig.autofmt_xdate()

plt.tight_layout()
plt.savefig(os.path.join(main_path, 'output', 'Graph4_STL.png'))




if DEBUG_MODE_ON:
        print("\n DEBUG ** Final modifications")
df_results['HISTORYDATE'] = df_results['HISTORYDATE'].dt.strftime('%Y-%m-%d')
df_graph1['HISTORYDATE'] = df_graph1['HISTORYDATE'].dt.strftime('%Y-%m-%d')
df_graph1 = df_graph1[df_graph1['PART_ID'] != 0]
df_graph2['HISTORYDATE'] = df_graph2['HISTORYDATE'].dt.strftime('%Y-%m-%d')
df_graph2 = df_graph2[df_graph2['PART_ID'] != 0]
df_graph3['HISTORYDATE'] = df_graph3['HISTORYDATE'].dt.strftime('%Y-%m-%d')
df_graph3 = df_graph3[df_graph3['PART_ID'] != 0]
df_graph4['HISTORYDATE'] = df_graph4['HISTORYDATE'].dt.strftime('%Y-%m-%d')
df_graph4 = df_graph4[df_graph4['PART_ID'] != 0]


if DEBUG_MODE_ON:
        print("\n DEBUG ** Writing excels outputs")
# Create the Excel file with the dataframe data
output_file = os.path.join(main_path, 'output', 'Calculo_Outliers_.xlsx')

with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    df_results.to_excel(writer, sheet_name='Hoja1', index=False)
    df_graph1.to_excel(writer, sheet_name='Graph1', index=False)
    df_graph2.to_excel(writer, sheet_name='Graph2', index=False)
    df_graph3.to_excel(writer, sheet_name='Graph3', index=False)
    df_graph4.to_excel(writer, sheet_name='Graph4', index=False)


print('Outlier calculation Excel generated')

##############################################################################################################################################################################
##############################################################################################################################################################################


t001 = time.time()

print("Outliers processed in " + str(int((t001 - t000))) + " s")

