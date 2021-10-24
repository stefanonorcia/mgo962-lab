from pandas import read_csv, DataFrame
from matplotlib import pyplot
from statsmodels.graphics.tsaplots import plot_acf

series = read_csv('ausbeer.csv', header=0, parse_dates=True, index_col=0, squeeze=True)

plot_acf(series, lags=20)
pyplot.show()
