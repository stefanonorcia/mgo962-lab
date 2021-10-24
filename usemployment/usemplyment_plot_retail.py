import pandas as pd
import matplotlib.pyplot as plt

# load the data
df = pd.read_csv("usemployment.csv", header=0, parse_dates=True, index_col=0)

dfs = df.query('Title == "Retail Trade"')
dfs = dfs.loc['1980-01-01':'2020-01-01']

dfs.plot()

plt.grid()
plt.show()
