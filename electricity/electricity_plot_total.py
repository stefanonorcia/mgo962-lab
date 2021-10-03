import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("electricity.csv", header=0, parse_dates=True, index_col=0)
df.plot()
plt.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
                mode="expand", borderaxespad=0, ncol=6)
plt.grid()
plt.show()
