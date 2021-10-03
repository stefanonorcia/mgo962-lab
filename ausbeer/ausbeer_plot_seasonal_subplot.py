import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("ausbeer.csv", header=0, parse_dates=True, index_col=0)
df['year'] = df.index.year
df['month'] = df.index.month
df = df.loc['1995-01-01':'2020-01-01']
months = df['month'].unique()
fig, axs = plt.subplots(1, 4, sharey=True)
for idx, month in enumerate(months):
    axs[idx].plot('Production', data=df.loc[df.month == month, :], label=month)
    axs[idx].grid()
plt.show()
