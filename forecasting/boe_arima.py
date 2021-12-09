from pandas import DataFrame
from io import StringIO
import time, json
from datetime import date
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_squared_error
from matplotlib.pylab import rcParams
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

rcParams['figure.figsize'] = 15, 6
bank_df = pd.read_csv('BOE-XUDLERD.csv')
print(bank_df)

# converting to time series data
bank_df['Date'] = pd.to_datetime(bank_df['Date'])
indexed_df = bank_df.set_index('Date')
ts = indexed_df['Value']
ts.head(5)

plt.plot(ts)
plt.show()

# Resample the data as it contains too much variations

ts_week = ts.resample('W').mean()
plt.plot(ts_week)


def test_stationarity(timeseries):
    # Determing rolling statistics
    rolmean = timeseries.rolling(window=52, center=False).mean()
    rolstd = timeseries.rolling(window=52, center=False).std()
    # Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue', label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    # Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print(dfoutput)


test_stationarity(ts_week)

# Since the test statistics is more than 5 % critical value and the p-value is larger than 0.05 ,
# the moving average is not constant over time and the null hypothesis of the Dickey-Fuller test cannot be rejected.
# This shows the weekly time series is not stationary.
# As such, we need to transform this series into a stationary time series.

ts_week_log = np.log(ts_week)
ts_week_log_diff = ts_week_log - ts_week_log.shift()
plt.plot(ts_week_log_diff)

# Again confirming with the dickey-fuller test

ts_week_log_diff.dropna(inplace=True)
test_stationarity(ts_week_log_diff)

# The test statistic is less than 1% of the critical value,
# shows that the time series is stationary with 99% confidence.

# ACF and PACF
lag_acf = acf(ts_week_log_diff, nlags=10)
lag_pacf = pacf(ts_week_log_diff, nlags=10, method='ols')

# Plot ACF:
plt.subplot(121)
plt.plot(lag_acf)
plt.axhline(y=0, linestyle='--', color='gray')
plt.axhline(y=-7.96 / np.sqrt(len(ts_week_log_diff)), linestyle='--', color='gray')
plt.axhline(y=7.96 / np.sqrt(len(ts_week_log_diff)), linestyle='--', color='gray')
plt.title('Autocorrelation Function')

# Plot PACF:
plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y=0, linestyle='--', color='gray')
plt.axhline(y=-7.96 / np.sqrt(len(ts_week_log_diff)), linestyle='--', color='gray')
plt.axhline(y=7.96 / np.sqrt(len(ts_week_log_diff)), linestyle='--', color='gray')
plt.title('Partial Autocorrelation Function')
plt.tight_layout()
plt.show()
# Using the plot we can determine the values for p and q respectively :
# p: the lag value where the PACF cuts off (drop to 0) for the first time. So here p=2.
# q: the lag value where the ACF chart crosses the upper confidence interval for the first time.
# if you look closely q=1.

# Optimal values fot ARIMA(p,d,q) model are (2,1,1). Hence plot the ARIMA model using the value (2,1,1)
model = ARIMA(ts_week_log, order=(2, 1, 1))
results_ARIMA = model.fit(disp=-1)
plt.plot(ts_week_log_diff)
plt.plot(results_ARIMA.fittedvalues, color='red')
plt.title('RSS: %.4f' % sum((results_ARIMA.fittedvalues - ts_week_log_diff) ** 2))
plt.show()

# print the results of the ARIMA model and plot the residuals

print(results_ARIMA.summary())
# plot residual errors
residuals = DataFrame(results_ARIMA.resid)
residuals.plot(kind='kde')
print(residuals.describe())

# Predictions

predictions_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues, copy=True)
print(predictions_ARIMA_diff.head())

# The model is returning the results we want to see, we can scale the model predictions back to the original scale.
# Hence the remove the first order differencing and take exponent
# to restore the predictions back to their original scale

predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
predictions_ARIMA_log = pd.Series(ts_week_log.iloc[0], index=ts_week_log.index)
predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum, fill_value=0)
predictions_ARIMA = np.exp(predictions_ARIMA_log)
plt.plot(ts_week)
plt.plot(predictions_ARIMA)
plt.title('RMSE: %.4f' % np.sqrt(sum((predictions_ARIMA - ts_week) ** 2) / len(ts_week)))

# Training and testing datsets
size = int(len(ts_week_log) - 15)
train, test = ts_week_log[0:size], ts_week_log[size:len(ts_week_log)]
history = [x for x in train]
predictions = list()

# Training the model and forecasting

size = int(len(ts_week_log) - 15)
train, test = ts_week_log[0:size], ts_week_log[size:len(ts_week_log)]
history = [x for x in train]
predictions = list()
print('Printing Predicted vs Expected Values...')
print('\n')
for t in range(len(test)):
    model = ARIMA(history, order=(2, 1, 1))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(float(yhat))
    obs = test[t]
    history.append(obs)
print('predicted=%f, expected=%f' % (np.exp(yhat), np.exp(obs)))

# Validating the model

error = mean_squared_error(test, predictions)
print('\n')
print('Printing Mean Squared Error of Predictions...')
print('Test MSE: %.6f' % error)
predictions_series = pd.Series(predictions, index=test.index)

# Plotting forecasted vs Observed values

fig, ax = plt.subplots()
ax.set(title='Spot Exchange Rate, Euro into USD', xlabel='Date', ylabel='Euro into USD')
ax.plot(ts_week[-60:], 'o', label='observed')
ax.plot(np.exp(predictions_series), 'g', label='rolling one-step out-of-sample forecast')
legend = ax.legend(loc='upper left')
legend.get_frame().set_facecolor('w')
plt.show()


