import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("electricity.csv", parse_dates=True, index_col=0)


df['week'] = df.index.weekofyear
df['day'] = df.index.dayofweek
weeks = df['week'].unique()
for week in weeks:
    dfWeek = df.loc[df.week == week, :]
    dfWeek['index'] = range(0, len(dfWeek))
    plt.plot('index', 'Demand', data=dfWeek, label=week)

plt.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
                mode="expand", borderaxespad=0, ncol=8)
plt.show()
