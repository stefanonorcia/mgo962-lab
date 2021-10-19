import pandas as pd
import matplotlib.pyplot as plt

# load the data
df = pd.read_csv("tourism.csv", parse_dates=True).drop(columns=["Unnamed: 0"])

# Total
dft = df.groupby("Quarter")
dft = dft['Trips'].agg(sum)
dft.plot()
plt.grid()
plt.show()