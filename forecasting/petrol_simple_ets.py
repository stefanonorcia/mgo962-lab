import pandas as pd
import numpy as np

import statsmodels.tsa.holtwinters as ets
import statsmodels.tools.eval_measures as fa
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from matplotlib import pyplot as plt


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

model = SimpleExpSmoothing(np.asarray(train['Consumption']))

alpha_list = [0.1, 0.5, 0.99]

pred_SES = test.copy()  # Have a copy of the test dataset

for alpha_value in alpha_list:
    alpha_str = "SES" + str(alpha_value)
    mode_fit_i = model.fit(smoothing_level=alpha_value, optimized=False)
    pred_SES[alpha_str] = mode_fit_i.forecast(len(test['Consumption']))
    rmse = np.sqrt(mean_squared_error(test['Consumption'], pred_SES[alpha_str]))
    mape = MAPE(test['Consumption'], pred_SES[alpha_str])
    ###
    print("For alpha = %1.2f,  RMSE is %3.4f MAPE is %3.2f" % (alpha_value, rmse, mape))
    plt.figure(figsize=(16, 8))
    plt.plot(train.TimeIndex, train['Consumption'], label='Train')
    plt.plot(test.TimeIndex, test['Consumption'], label='Test')
    plt.plot(test.TimeIndex, pred_SES[alpha_str], label=alpha_str)
    plt.title('Simple Exponential Smoothing with alpha ' + str(alpha_value))
    plt.legend(loc='best')
    plt.show()

pred_opt = SimpleExpSmoothing(train['Consumption']).fit(optimized=True)
print('')
print('== Simple Exponential Smoothing ')
print('')

print('')
print('Smoothing Level', np.round(pred_opt.params['smoothing_level'], 4))
print('Initial Level', np.round(pred_opt.params['initial_level'], 4))
print('')

y_pred_opt = pred_opt.forecast(steps=16)
df_pred_opt = pd.DataFrame({'Y_hat': y_pred_opt, 'Y': test['Consumption'].values})

rmse_opt = np.sqrt(mean_squared_error(test['Consumption'], y_pred_opt))
mape_opt = MAPE(test['Consumption'], y_pred_opt)

alpha_value = np.round(pred_opt.params['smoothing_level'], 4)

print("For alpha = %1.2f,  RMSE is %3.4f MAPE is %3.2f" % (alpha_value, rmse_opt, mape_opt))

plt.figure(figsize=(16, 8))
plt.plot(train.TimeIndex, train['Consumption'], label='Train')
plt.plot(test.TimeIndex, test['Consumption'], label='Test')
plt.plot(test.TimeIndex, y_pred_opt, label='SES_OPT')
plt.title('Simple Exponential Smoothing with alpha ' + str(alpha_value))
plt.legend(loc='best')
plt.show()

print(df_pred_opt.head().T)
