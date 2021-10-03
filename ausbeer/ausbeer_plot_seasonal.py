import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("ausbeer.csv", header=0, parse_dates=True, index_col=0)
df['year'] = df.index.year
df['month'] = df.index.month

df = df.loc['1995-01-01':'2020-01-01']

years = df['year'].unique()

for year in years:
    plt.plot('month', 'Production', data=df.loc[df.year == year, :], label=year)

plt.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
                mode="expand", borderaxespad=0, ncol=6)
plt.grid()
plt.show()
