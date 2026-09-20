# Outlier Detection for Demand Analysis

## Overview

This script was designed to detect abnormal values in historical demand data for different materials or parts. It combines two complementary outlier detection techniques:

- Z-score analysis
- STL decomposition (Seasonal-Trend decomposition using LOESS)

The goal is to identify unusually high or low demand points that may be caused by data-entry issues, exceptional events, or operational anomalies.

The script is intended for demand and planning analysis scenarios where time series are grouped by material and date, and where values must be reviewed before forecasting or replenishment decisions.

---

## What the script does

The workflow is as follows:

1. Loads the input datasets from local files and optional database sources.
2. Normalizes raw data fields such as document references, part identifiers, dates, positions, and quantity values.
3. Merges the main data set with a comparison dataset using keys such as document and position.
4. Aggregates values by part and date.
5. Applies a Z-score check to detect statistically extreme values.
6. Applies STL decomposition to identify anomalies in residuals after removing trend and seasonality.
7. Combines both methods to classify candidate outliers.
8. Generates adjusted values and cap/winsorization logic for suspect records.
9. Produces visual charts and an Excel file with processed results.

---

## Main objective

The script answers the following business question:

> Which historical demand values are abnormal enough to deserve review, correction, or exclusion before forecasting or stock planning?

By using both statistical and time-series-based methods, it reduces false positives and helps analysts focus on the records that most likely require intervention.

---

## Input data

The script expects one or more structured datasets, especially:

- part identifier (`PARTID`)
- date (`HISTORYDATE`)
- quantity (`QTY`)
- document number (`DOCUMENT`)
- order reason (`ORDER_REASON`)
- document type (`TIPO_DOCUMENTO`)
- position

It relies on helper modules such as:

- `df_creator_module`
- `db_creator_reader_module`

The script checks whether the data should be loaded from files or from SQLite databases depending on the `CHECK_UPDATE` flag.

---

## Data normalization

Before analysis, the script cleans and standardizes the raw tables:

- removes empty rows
- strips strings and whitespace
- converts date strings to proper pandas timestamps
- ensures numeric columns are parsed correctly
- extracts document and position information from text-based identifiers
- filters invalid records and blank values

This step is critical because the analysis depends on consistently formatted keys and quantities.

---

## Outlier detection logic

### 1. Z-score method

For each part history, the script computes the Z-score of the quantity by comparing each measurement against the series distribution.

Parameters used:

- `Z_THRESHOLD = 3.0`
- `manual_season = ['A', 'B', 'C']`

A value is marked as an outlier when it exceeds the threshold in absolute value. The script also computes an adjustment value for flagged high outliers.

### 2. STL method

The script also applies STL decomposition on the quantity series, using a seasonal period defined by:

- `STL_PERIOD = 365`

This helps separate:

- trend
- seasonality
- residuals

Residuals outside the expected range are flagged as outliers. This allows the detection of anomalies that are not simply high variance points, but values deviating from the underlying temporal pattern.

### 3. Combined logic

The final result is not based on a single metric only. If either method identifies a row as an outlier, it is treated as a candidate anomaly and is included in the combined output.

The script also calculates secondary fields such as:

- `Z_ADJUST`
- `STL_ADJUST`
- `Winsorization`
- `Min-adjust`
- `Max-capping`
- `Med`
- `LOG`

These values are used to evaluate how aggressive the adjustment should be when a suspicious value is corrected or capped.

---

## How the data is processed

The script groups records by `PARTID` and `HISTORYDATE`, then aggregates values for each date.

It does the following per material:

- sorts data by date
- removes null quantities
- computes the Z-score
- evaluates STL residuals
- marks potential outliers
- builds adjusted/capped values for reporting

---

## Debug and testing mode

The variable `DEBUG_MODE_ON` enables detailed output during execution.

When enabled, the script prints:

- normalization diagnostics
- Z-score analysis status
- STL analysis status
- summary statistics by material
- chart-generation status

This is useful during development, testing, and validation of result quality.

---

## Outputs generated

The script creates an output folder and writes the following files:

### Excel file

- `Calculo_Outliers_.xlsx`

This contains the processed results, including outlier flags and adjusted values for all analyzed rows.

### Charts

The script generates several PNG visualizations for representative materials, such as:

- `Graph1_ZScore.png`
- `Graph1_STL.png`
- `Graph2_ZScore.png`
- `Graph2_STL.png`
- `Graph3_ZScore.png`
- `Graph3_STL.png`
- `Graph4_ZScore.png`
- `Graph4_STL.png`

These graphs help validate the outlier detection visually by showing the original demand series, trend, seasonality, and residual behavior.

---

## Configuration parameters

The script includes several adjustable parameters at the top of the file:

- `FILE_DATE_INPUT`: date used to process the dataset
- `RESAMPLE_RULE`: frequency for resampling (daily, weekly, monthly)
- `MIN_POINTS`: minimum observations per series
- `Z_THRESHOLD`: threshold for Z-score anomaly detection
- `STL_THRESHOLD`: threshold for residual anomaly detection
- `STL_PERIOD`: STL seasonal period
- `manual_season`: list of reasons treated as manually seasonal
- `USE_ROBUST`: option for robust calculations
- `DEBUG_MODE_ON`: enables extra logging and generation of intermediate outputs

---

## Dependencies

This project requires Python libraries such as:

- pandas
- numpy
- matplotlib
- statsmodels
- scipy

It also depends on custom project modules for reading source data and local database access.

---

## Execution flow

When the script runs, it does the following:

1. Sets the working directory to the script folder.
2. Adds custom utility folders to `sys.path`.
3. Loads the input data source.
4. Normalizes the data.
5. Merges the comparison dataset.
6. Groups by material and date.
7. Executes the Z-score detection.
8. Executes STL anomaly detection.
9. Builds summary tables and charts.
10. Saves the final Excel workbook and plots.

---

## Important notes

- The script is tailored to a specific business context: inventory/demand anomaly review.
- Some paths and names are hardcoded and may require updates depending on the repository or execution environment.
- The use of `rute/` placeholders suggests the project expects local folders or integration paths specific to the original environment.
- The script currently focuses on a few selected part IDs for chart generation, so the graphs are examples rather than exhaustive material coverage.

---

## Use cases

This script is useful for:

- anomaly detection in historical demand data
- review of suspicious purchase or consumption spikes
- data-quality validation for planning systems
- preparing clean data before forecasting models
- identifying outliers at material level across time

---

## Summary

In short, this script is an automated anomaly detection pipeline for historical demand series. It combines statistical methods and time-series decomposition to identify abnormal values, generate flags, calculate adjustment estimates, and export the results to Excel and visual plots.

It is designed to support analysts and planners who need to review unusual demand patterns before forecasting, procurement, or inventory management decisions.
