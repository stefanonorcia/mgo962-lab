import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf

# load the data
df = pd.read_csv("tourism.csv", parse_dates=True).drop(columns=["Unnamed: 0"])

# Total
dft = df.groupby("Quarter")
dft = dft['Trips'].agg(sum)
plot_acf(dft, lags=20)

plt.grid()
plt.show()
