import pandas as pd
import matplotlib.pyplot as plt

# load the data
df = pd.read_csv("aus_production.csv", header=0, parse_dates=['Quarter'], index_col=0)

# Total
plt.plot('Electricity', data=df)
plt.grid()
plt.show()
