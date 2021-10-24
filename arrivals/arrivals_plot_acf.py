import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf

# load the data
df = pd.read_csv("arrivals.csv", parse_dates=True)

# Total
dft = df.groupby("index")
dft = dft['value'].agg(sum)
plot_acf(dft, lags=20)
plt.show()
