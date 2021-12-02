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


AirPax = pd.read_csv('passengers.csv')
date_rng = pd.date_range(start='1/1/1949', end='31/12/1960', freq='M')
print(date_rng)

AirPax['TimeIndex'] = pd.DataFrame(date_rng, columns=['Month'])
print(AirPax.head())

train = AirPax[0:int(len(AirPax)*0.7)]
test = AirPax[int(len(AirPax)*0.7):]

print("\n Training data start at \n")
print(train[train.TimeIndex == train.TimeIndex.min()], ['Year', 'Month'])
print("\n Training data ends at \n")
print(train[train.TimeIndex == train.TimeIndex.max()], ['Year', 'Month'])

print("\n Test data start at \n")
print(test[test.TimeIndex == test.TimeIndex.min()], ['Year', 'Month'])

print("\n Test data ends at \n")
print(test[test.TimeIndex == test.TimeIndex.max()], ['Year', 'Month'])

plt.plot(train.TimeIndex, train.Passenger, label='Train')
plt.plot(test.TimeIndex, test.Passenger,  label='Test')
plt.legend(loc='best')
plt.title('Original data after split')
plt.show()

pred = ExponentialSmoothing(np.asarray(train['Passenger']),
                            trend='multiplicative',
                            seasonal_periods=12,
                            seasonal='multiplicative').fit(optimized=True)

print(pred.params)

print('')
print('== Holt-Winters Additive ETS(A,A,A) Parameters ==')
print('')
alpha_value = np.round(pred.params['smoothing_level'], 4)
print('Smoothing Level: ', alpha_value)
print('Smoothing Seasonal: ', np.round(pred.params['smoothing_seasonal'], 4))
print('Initial Level: ', np.round(pred.params['initial_level'], 4))
print('Initial Seasons: ', np.round(pred.params['initial_seasons'], 4))
print('')

pred_HoltW = test.copy()
pred_HoltW['HoltW'] = pred.forecast(len(test['Passenger']))
plt.figure(figsize=(16, 8))
plt.plot(train['Passenger'], label='Train')
plt.plot(test['Passenger'], label='Test')
plt.plot(pred_HoltW['HoltW'], label='HoltWinters')
plt.title('Holt-Winters Additive ETS(A,A,A) Parameters:\n  alpha = ' +
          str(alpha_value) + '  Gamma: ' + str(np.round(pred.params['smoothing_seasonal'], 4)))
plt.legend(loc='best')
plt.show()

df_pred_opt = pd.DataFrame({'Y_hat': pred_HoltW['HoltW'], 'Y': test['Passenger'].values})

rmse_opt = np.sqrt(mean_squared_error(df_pred_opt.Y, df_pred_opt.Y_hat))
mape_opt = MAPE(df_pred_opt.Y, df_pred_opt.Y_hat)

print("For alpha = %1.2f,  RMSE is %3.4f MAPE is %3.2f" %(alpha_value, rmse_opt, mape_opt))

