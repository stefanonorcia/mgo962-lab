import pandas as pd
import matplotlib.pyplot as plt

# load the data
df = pd.read_csv("PBS.csv", header=0, parse_dates=['Month'], index_col=0)

dft = df.groupby("Month")
dft = dft['Cost'].agg(sum)

# Total
dft.plot()
plt.grid()
plt.show()
