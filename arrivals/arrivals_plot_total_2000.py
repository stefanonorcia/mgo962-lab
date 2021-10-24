import pandas as pd
import matplotlib.pyplot as plt

# load the data
df = pd.read_csv("arrivals.csv", parse_dates=True)

# Total
dft = df.groupby("index")
dft = dft['value'].agg(sum)
dft = dft.loc['2000-01-01':'2020-01-01']

dft.plot()
plt.grid()
plt.show()
