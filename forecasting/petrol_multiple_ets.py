import pandas as pd
import numpy as np

import statsmodels.tsa.holtwinters as ets
import statsmodels.tools.eval_measures as fa
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from matplotlib import pyplot as plt
from statsmodels.tsa.holtwinters import Holt


def MAPE(y, yhat):
    y, yhat = np.array(y), np.array(yhat)
    try:
        mape = round(np.sum(np.abs(yhat - y)) / np.sum(y) * 100, 2)
    except:
        print("Observed values are empty")
        mape = np.nan
    return mape


petrol_df = pd.read_csv('Petrol.csv')
date_rng = pd.date_range(start='1/1/2001', end='30/9/2013', freq='Q')
print(date_rng)

petrol_df['TimeIndex'] = pd.DataFrame(date_rng, columns=['Quarter'])
print(petrol_df.head(3).T)

plt.plot(petrol_df.TimeIndex, petrol_df.Consumption)
plt.title('Original data before split')
plt.show()

train = petrol_df[0:int(len(petrol_df) * 0.7)]
test = petrol_df[int(len(petrol_df) * 0.7):]

print("\n Training data start at \n")
print(train[train.TimeIndex == train.TimeIndex.min()], ['Year', 'Quarter'])
print("\n Training data ends at \n")
print(train[train.TimeIndex == train.TimeIndex.max()], ['Year', 'Quarter'])

print("\n Test data start at \n")
print(test[test.TimeIndex == test.TimeIndex.min()], ['Year', 'Quarter'])

print("\n Test data ends at \n")
print(test[test.TimeIndex == test.TimeIndex.max()], ['Year', 'Quarter'])

plt.plot(train.TimeIndex, train.Consumption, label='Train')
plt.plot(test.TimeIndex, test.Consumption, label='Test')
plt.legend(loc='best')
plt.title('Original data after split')
plt.show()

pred1 = ExponentialSmoothing(np.asarray(train['Consumption']),
                             trend='multiplicative',
                             seasonal='multiplicative',
                             seasonal_periods=12).fit()

print('')
print('== Holt-Winters Additive ETS(A,A,A) Parameters ==')
print('')
alpha_value = np.round(pred1.params['smoothing_level'], 4)
print('Smoothing Level: ', alpha_value)
# print('Smoothing Slope: ', np.round(pred1.params['smoothing_slope'], 4))
print('Smoothing Seasonal: ', np.round(pred1.params['smoothing_seasonal'], 4))
print('Initial Level: ', np.round(pred1.params['initial_level'], 4))
# print('Initial Slope: ', np.round(pred1.params['initial_slope'], 4))
print('Initial Seasons: ', np.round(pred1.params['initial_seasons'], 4))
print('')

### Forecast for next 16 months

y_pred1 = pred1.forecast(steps=16)
df_pred1 = pd.DataFrame({'Y_hat': y_pred1, 'Y': test['Consumption']})
print(df_pred1)

### Plot

fig2, ax = plt.subplots()
ax.plot(df_pred1.Y, label='Original')
ax.plot(df_pred1.Y_hat, label='Predicted')

plt.legend(loc='upper left')
plt.title('Holt-Winters Additive ETS(A,A,A) Method 1')
plt.ylabel('Qty')
plt.xlabel('Date')
plt.show()

rmse = np.sqrt(mean_squared_error(df_pred1.Y, df_pred1.Y_hat))
mape = MAPE(df_pred1.Y, df_pred1.Y_hat)

print("For alpha = %1.2f,  RMSE is %3.4f MAPE is %3.2f" %(alpha_value, rmse, mape))

print(pred1.params)



