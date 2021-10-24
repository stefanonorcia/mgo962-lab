import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf

# load the data
df = pd.read_csv("usemployment.csv", header=0, parse_dates=True, index_col=0)

dfs = df.query('Title == "Retail Trade"')
dfs = dfs.loc['1980-01-01':'2020-01-01']

fig, axs = plt.subplots(2, 5, sharey=True)
pd.plotting.lag_plot(dfs['Employed'], lag=1, ax=axs[0][0])
pd.plotting.lag_plot(dfs['Employed'], lag=2, ax=axs[0][1])
pd.plotting.lag_plot(dfs['Employed'], lag=3, ax=axs[0][2])
pd.plotting.lag_plot(dfs['Employed'], lag=4, ax=axs[0][3])
pd.plotting.lag_plot(dfs['Employed'], lag=5, ax=axs[0][4])
pd.plotting.lag_plot(dfs['Employed'], lag=6, ax=axs[1][0])
pd.plotting.lag_plot(dfs['Employed'], lag=7, ax=axs[1][1])
pd.plotting.lag_plot(dfs['Employed'], lag=8, ax=axs[1][2])
pd.plotting.lag_plot(dfs['Employed'], lag=9, ax=axs[1][3])
pd.plotting.lag_plot(dfs['Employed'], lag=10, ax=axs[1][4])

plt.grid()
plt.show()
