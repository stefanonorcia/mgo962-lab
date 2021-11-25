import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.metrics import r2_score
import statsmodels.api as sm


import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns


dataset = pd.read_csv('petrol_consumption.csv')
print(dataset.head())

plt.figure(figsize=(10, 6))
sns.heatmap(dataset.corr(), cmap=plt.cm.Reds, annot=True)

plt.show()

X = dataset[['Petrol_tax', 'Average_income', 'Paved_Highways', 'Population_Driver_licence(%)']]
y = dataset['Petrol_Consumption']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

regressor = LinearRegression()
regressor.fit(X_train, y_train)

coeff_df = pd.DataFrame(regressor.coef_, X.columns, columns=['Coefficient'])
int_df = pd.DataFrame(regressor.intercept_, X.columns, columns=['Coefficient'])
print(coeff_df)
print(int_df)

y_pred = regressor.predict(X_test)

df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

plt.plot(df['Actual'], "o", color="blue")
plt.show()

print(df)

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print("Coefficient of determination: %.2f" % r2_score(y_test, y_pred))


X_train = sm.add_constant(X_train)

model = sm.OLS(y_train, X_train).fit()
predictions = model.predict(X_train)

print_model = model.summary()
print(print_model)
