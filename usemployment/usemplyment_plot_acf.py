import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf

# load the data
df = pd.read_csv("usemployment.csv", header=0, parse_dates=True, index_col=0)

dfs = df.query('Title == "Retail Trade"')
dfs = dfs.loc['1980-01-01':'2020-01-01']
plot_acf(dfs['Employed'], lags=48)

plt.grid()
plt.show()
