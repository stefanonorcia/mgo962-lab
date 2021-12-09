import calendar
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.stattools import adfuller
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

data = pd.read_csv('TractorSales.csv', header=0, parse_dates=[0], squeeze=True)

dates = pd.date_range(start='2003-01-01', freq='MS', periods=len(data))

data['Month'] = dates.month
data['Month'] = data['Month'].apply(lambda x: calendar.month_abbr[x])
data['Year'] = dates.year

data.drop(['Month-Year'], axis=1, inplace=True)
data.rename(columns={'Number of Tractor Sold': 'Tractor-Sales'}, inplace=True)

data = data[['Month', 'Year', 'Tractor-Sales']]
data.set_index(dates, inplace=True)

sales_ts = data['Tractor-Sales']

result = adfuller(sales_ts)

print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])

sales_ts_diff = sales_ts - sales_ts.shift(periods=1)
sales_ts_diff.dropna(inplace=True)

result = adfuller(sales_ts_diff)

pval = result[1]
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])

if pval < 0.05:
    print('Data is stationary')
else:
    print('Data after differencing is not stationary; so try log diff')
    sales_ts_log = np.log10(sales_ts)
    sales_ts_log.dropna(inplace=True)
    sales_ts_log_diff = sales_ts_log.diff(periods=1)
    sales_ts_log_diff.dropna(inplace=True)
    result = adfuller(sales_ts_log_diff)

    pval = result[1]
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    if pval < 0.05:
        print('Data after log differencing is stationary')
    else:
        print('Data after log differencing is not stationary; try second order differencing')
        sales_ts_log_diff2 = sales_ts_log.diff(periods=2)
        sales_ts_log_diff2.dropna(inplace=True)
        result = adfuller(sales_ts_log_diff2)
        pval = result[1]
        print('ADF Statistic: %f' % result[0])
        print('p-value: %f' % result[1])
        if pval < 0.05:
            print('Data after log differencing 2nd order is stationary')
        else:
            print('Data after log differencing 2nd order is not stationary')

lag_acf = acf(sales_ts_log_diff2, nlags=20)
lag_pacf = pacf(sales_ts_log_diff2, nlags=20, method='ols')

# Plot ACF:

plt.figure(figsize=(15, 5))
plt.subplot(121)
plt.stem(lag_acf)
plt.axhline(y=0, linestyle='--', color='black')
plt.axhline(y=-1.96 / np.sqrt(len(sales_ts_log_diff2)), linestyle='--', color='gray')
plt.axhline(y=1.96 / np.sqrt(len(sales_ts_log_diff2)), linestyle='--', color='gray')
plt.xticks(range(0, 22, 1))
plt.xlabel('Lag')
plt.ylabel('ACF')
plt.title('Autocorrelation Function')

# Plot PACF:

plt.subplot(122)
plt.stem(lag_pacf)
plt.axhline(y=0, linestyle='--', color='black')
plt.axhline(y=-1.96 / np.sqrt(len(sales_ts_log_diff2)), linestyle='--', color='gray')
plt.axhline(y=1.96 / np.sqrt(len(sales_ts_log_diff2)), linestyle='--', color='gray')
plt.xlabel('Lag')
plt.xticks(range(0, 22, 1))
plt.ylabel('PACF')
plt.title('Partial Autocorrelation Function')
plt.tight_layout()

plt.show()
