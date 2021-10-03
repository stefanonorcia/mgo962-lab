import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("tourism.csv", parse_dates=True, index_col=1).drop(columns=["Unnamed: 0"])

df.columns = [col_name.lower() for col_name in df.columns]
df = df.loc['2000-01-01':'2016-01-01']

fig, axs = plt.subplots(7, 1)

df['quarter'] = pd.DatetimeIndex(df.index)
df['year'] = pd.DatetimeIndex(df['quarter']).year
df = df.query('purpose == "Holiday"')

states = df['state'].unique()
years = df['year'].unique()
for idx, state in enumerate(states):
    for idy, year in enumerate(years):
        dfs = df.query('state == "' + state + '"')
        dfs = dfs.loc[str(year) + '-01-01': str(year) + '-12-31']
        dfy = dfs.groupby('quarter', as_index=False)
        dfy = dfy['trips'].agg(sum)
        dfy['month'] = pd.DatetimeIndex(dfy['quarter']).month
        dfy = dfy.sort_values(by=['month'], ascending=True)
        axs[idx].plot('month', 'trips', data=dfy, label=year)
        axs[idx].title.set_text(state)

plt.subplots_adjust(hspace=0.5)
plt.grid()
plt.show()

