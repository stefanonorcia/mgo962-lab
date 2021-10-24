import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# load the data
df = pd.read_csv("usemployment.csv", header=0, parse_dates=True, index_col=0)

dfs = df.query('Title == "Retail Trade"')
dfs = dfs.loc['1980-01-01':'2020-01-01']
decomposition = seasonal_decompose(dfs['Employed'], model='additive')
fig = decomposition.plot()
plt.show()
