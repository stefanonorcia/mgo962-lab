import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf

# load the data
df = pd.read_csv("tourism.csv",
                 parse_dates=True).drop(columns=["Unnamed: 0"])
# Total
dft = df.groupby("Quarter")
dft = dft['Trips'].agg(sum)
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

plt.grid()
plt.show()
