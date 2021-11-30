import pandas as pd
import numpy as np

import statsmodels.tsa.holtwinters as ets
import statsmodels.tools.eval_measures as fa
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
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

model = Holt(np.asarray(train['Consumption']))

model_fit = model.fit(smoothing_level=0.8, smoothing_slope=0.2, optimized=False)

print('')
print('==Holt model Exponential Smoothing Parameters ==')
print('')
alpha_value = np.round(model_fit.params['smoothing_level'], 4)
print('Smoothing Level', alpha_value)
# print('Smoothing Slope', np.round(model_fit.params['smoothing_slope'], 4))
print('Initial Level', np.round(model_fit.params['initial_level'], 4))
print('')

Pred_Holt = test.copy()
Pred_Holt['Opt'] = model_fit.forecast(len(test['Consumption']))

plt.figure(figsize=(16, 8))
plt.plot(train['Consumption'], label='Train')
plt.plot(test['Consumption'], label='Test')
plt.plot(Pred_Holt['Opt'], label='HoltOpt')
plt.legend(loc='best')
plt.show()

df_pred_opt = pd.DataFrame({'Y_hat': Pred_Holt['Opt'], 'Y': test['Consumption'].values})
rmse_opt = np.sqrt(mean_squared_error(df_pred_opt.Y, df_pred_opt.Y_hat))
mape_opt = MAPE(df_pred_opt.Y, df_pred_opt.Y_hat)

print("For alpha = %1.2f,  RMSE is %3.4f MAPE is %3.2f" % (alpha_value, rmse_opt, mape_opt))

print(model_fit.params)
