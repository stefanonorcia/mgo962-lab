import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# load the data
df = pd.read_csv("tourism.csv", parse_dates=True).drop(columns=["Unnamed: 0"])

# Total
dft = df.groupby("Quarter")
dft = dft['Trips'].agg(sum)
decomposition = seasonal_decompose(dft, model='additive', period=4)
fig = decomposition.plot()

plt.grid()
plt.show()
