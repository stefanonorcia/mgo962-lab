import pandas as pd
from matplotlib import pyplot
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.naive import NaiveForecaster

from sktime.datasets import load_airline
from sktime.forecasting.model_selection import temporal_train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from math import sqrt

y = load_airline()

print(y.head())

y_train, y_test = temporal_train_test_split(y)

fh = ForecastingHorizon(
    pd.PeriodIndex(pd.date_range("1958-01", periods=36, freq="M")), is_relative=False
)

y_train.plot(figsize=(15, 8),  fontsize=14)
y_test.plot(figsize=(15, 8),  fontsize=14)

naive_forecaster_last = NaiveForecaster(strategy="last")
naive_forecaster_last.fit(y_train)
y_last = naive_forecaster_last.predict(fh)
y_last.plot(figsize=(15, 8), label='naive')

mae = mean_absolute_error(y_test, y_last)
mse = mean_squared_error(y_test, y_last)
mape = mean_absolute_percentage_error(y_test, y_last)

print(mae)
print(mse)
print(mape)
print(sqrt(mse))

naive_forecaster_mean = NaiveForecaster(strategy="mean")
naive_forecaster_mean.fit(y_train)
y_mean = naive_forecaster_mean.predict(fh)
y_mean.plot(figsize=(15, 8), label='mean')

# naive_forecaster_drift = NaiveForecaster(strategy="drift")
# naive_forecaster_drift.fit(y_train)
# y_drift = naive_forecaster_drift.predict(fh)
# y_drift.plot(figsize=(15, 8), label='drift')

# naive_forecaster_last_sp = NaiveForecaster(strategy="last", sp=12)
# naive_forecaster_last_sp.fit(y_train)
# y_last_sp = naive_forecaster_last_sp.predict(fh)
# y_last_sp.plot(figsize=(15, 8), label='snaive')

pyplot.legend()
pyplot.grid()
pyplot.show()
