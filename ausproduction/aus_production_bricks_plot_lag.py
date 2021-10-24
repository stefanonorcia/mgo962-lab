import pandas as pd
import matplotlib.pyplot as plt

# load the data
df = pd.read_csv("aus_production.csv",
                 header=0, parse_dates=True, index_col=0)
dfs = df.loc['1956 Q1':'2005 Q2']
fig, axs = plt.subplots(2, 4, sharey=True)

pd.plotting.lag_plot(dfs, lag=1, ax=axs[0][0])
pd.plotting.lag_plot(dfs, lag=2, ax=axs[0][1])
pd.plotting.lag_plot(dfs, lag=3, ax=axs[0][2])
pd.plotting.lag_plot(dfs, lag=4, ax=axs[0][3])
pd.plotting.lag_plot(dfs, lag=5, ax=axs[1][0])
pd.plotting.lag_plot(dfs, lag=6, ax=axs[1][1])
pd.plotting.lag_plot(dfs, lag=7, ax=axs[1][2])
pd.plotting.lag_plot(dfs, lag=8, ax=axs[1][3])
plt.show()
