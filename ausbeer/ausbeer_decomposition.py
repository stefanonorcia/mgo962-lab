import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

df = pd.read_csv("ausbeer.csv",  header=0, parse_dates=True, index_col=0)

decomposition = seasonal_decompose(df, model='additive')
fig = decomposition.plot()
plt.show()
