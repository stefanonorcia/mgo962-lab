import pandas as pd
import matplotlib.pyplot as plt

# load the data
df = pd.read_csv("aus_production.csv", header=0, parse_dates=['Quarter'], index_col=0)
dfs = df.loc['1956-01-01':'2005-01-01']

# Total
plt.plot('Bricks', data=df)
plt.grid()
plt.show()
