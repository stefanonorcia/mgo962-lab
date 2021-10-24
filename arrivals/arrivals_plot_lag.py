import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv("arrivals.csv",
                 header=0, parse_dates=True, index_col=0)
dft = df.groupby("index")
dft = dft['value'].agg(sum)
dft = dft.loc['2000-01-01':'2020-01-01']
fig, axs = plt.subplots(2, 5, sharey=True)
pd.plotting.lag_plot(dft, lag=1, ax=axs[0][0])
pd.plotting.lag_plot(dft, lag=2, ax=axs[0][1])
pd.plotting.lag_plot(dft, lag=3, ax=axs[0][2])
pd.plotting.lag_plot(dft, lag=4, ax=axs[0][3])
pd.plotting.lag_plot(dft, lag=5, ax=axs[0][4])
pd.plotting.lag_plot(dft, lag=6, ax=axs[1][0])
pd.plotting.lag_plot(dft, lag=7, ax=axs[1][1])
pd.plotting.lag_plot(dft, lag=8, ax=axs[1][2])
pd.plotting.lag_plot(dft, lag=9, ax=axs[1][3])
pd.plotting.lag_plot(dft, lag=10, ax=axs[1][4])
plt.show()
