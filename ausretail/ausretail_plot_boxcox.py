import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# load the data
df = pd.read_csv("aus_retail.csv",
                 header=0, parse_dates=True, index_col=3)

dfs = df.query('Industry == "Food retailing"')
dfs = dfs.loc['1990-01-01':'2020-01-01']

fig, axs = plt.subplots(2, 2)

dfs['Log'] = np.log10(dfs['Turnover'])
dfs['Sqr'] = np.sqrt(dfs['Turnover'])
dfs['Inv'] = np.subtract(1, dfs['Turnover'])

dfs = dfs.groupby("Month")

dft = dfs['Turnover'].agg(sum)
axs[0][0].set_title("Turnover")
axs[0][0].plot(dft)
axs[0][0].grid()

dfl = dfs['Log'].agg(sum)
axs[0][1].set_title("Log")
axs[0][1].plot(dfl, label='Log')
axs[0][1].grid()

dfsq = dfs['Sqr'].agg(sum)
axs[1][1].set_title("Sqr")
axs[1][1].plot(dfsq, label='Sqr')
axs[1][1].grid()

dfi = dfs['Inv'].agg(sum)
axs[1][0].set_title("Inv")
axs[1][0].plot(dfi, label='Inv')
axs[1][0].grid()

plt.show()
