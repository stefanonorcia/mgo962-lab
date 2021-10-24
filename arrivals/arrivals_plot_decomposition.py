import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
# load the data
df = pd.read_csv("arrivals.csv", parse_dates=True)
# Total
dft = df.groupby("index")
dft = dft['value'].agg(sum)
decomposition = seasonal_decompose(dft, model='additive', period=4)
fig = decomposition.plot()
plt.show()
