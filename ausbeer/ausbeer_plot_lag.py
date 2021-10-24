import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("ausbeer.csv", header=0, parse_dates=True, index_col=0)
df = df.loc['1992-01-01':'2020-01-01']
fig, axs = plt.subplots(2, 4, sharey=True)
pd.plotting.lag_plot(df, lag=1, ax=axs[0][0])
pd.plotting.lag_plot(df, lag=2, ax=axs[0][1])
pd.plotting.lag_plot(df, lag=3, ax=axs[0][2])
pd.plotting.lag_plot(df, lag=4, ax=axs[0][3])
pd.plotting.lag_plot(df, lag=5, ax=axs[1][0])
pd.plotting.lag_plot(df, lag=6, ax=axs[1][1])
pd.plotting.lag_plot(df, lag=7, ax=axs[1][2])
pd.plotting.lag_plot(df, lag=8, ax=axs[1][3])
plt.show()
