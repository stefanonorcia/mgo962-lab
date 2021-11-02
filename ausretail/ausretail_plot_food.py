import pandas as pd
import matplotlib.pyplot as plt

# load the data
df = pd.read_csv("aus_retail.csv",
                 header=0, parse_dates=True, index_col=3)

dfs = df.query('Industry == "Food retailing"')
dfs = dfs.loc['1990-01-01':'2020-01-01']
dfs = dfs.groupby("Month")
dfs = dfs['Turnover'].agg(sum)

dfs.plot()

plt.grid()
plt.show()
