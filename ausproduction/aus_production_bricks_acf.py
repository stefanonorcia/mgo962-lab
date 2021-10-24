import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf

# load the data
df = pd.read_csv("aus_production.csv", header=0, parse_dates=['Quarter'], index_col=0)
dfs = df.loc['1956 Q1':'2005 Q2']
plot_acf(dfs['Bricks'], lags=48)

# Total
plt.grid()
plt.show()
