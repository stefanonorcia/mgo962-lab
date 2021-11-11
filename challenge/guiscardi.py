# STUDENT: Guiscardi Alessandro

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('electricity.csv', parse_dates= True)
print(df.head())

# Dataset preparation
df['Time'] = pd.to_datetime(df['Time'])
df['Day'] = df['Time'].dt.weekday
df['Year'] = df['Time'].dt.year
df['Week'] = df['Time'].dt.week
df['Hour'] = df['Time'].dt.hour
df = df.loc[df['Year'] >= 2012]
print(df.head())

# Yearly demand with quarterly frequency
df['week_day'] = df['Week'].astype(str) + '' + df['Day'].astype(str)

sns.lineplot(x='week_day', y='Demand',  hue='Year', estimator=None,  data=df).set(title = 'Plot_1', xticklabels=[], xlabel='Time')
plt.show()

# Weekly demand with daily intervals
df['week_year'] = df['Year'].astype(str) + ''+ df['Week'].astype(str)
df['day_hour'] = df['Day'].astype(str) + ''+ df['Hour'].astype(str)
sns.lineplot(x='day_hour', y='Demand', hue='week_year', estimator=None, data=df,
             legend=False).set(title='Plot_2', xlabel = 'Time', xticklabels=[])
plt.show()

# Daily demand with hourly intervals
sns.lineplot(x='Hour', y='Demand', hue='Date', estimator=None, data=df, legend = False).set(title='Plot_3', xlabel='Time')
plt.show()