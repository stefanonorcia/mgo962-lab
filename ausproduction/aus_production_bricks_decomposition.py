import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# load the data
df = pd.read_csv("aus_production.csv", header=0, parse_dates=True, index_col=0)
dfs = df.loc['1956 Q1':'2004 Q2']
# Total
decomposition = seasonal_decompose(dfs['Bricks'], model='additive', period=4)
fig = decomposition.plot()
plt.show()