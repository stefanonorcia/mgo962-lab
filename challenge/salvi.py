# CHALLENGE 1 TIME SERIES ELISA SALVI
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from random import randrange
from pandas import Series
import seaborn as sns
from matplotlib import pyplot
from datetime import datetime
from statsmodels.tsa.seasonal import seasonal_decompose

vic_elec = pd.read_csv('electricity.csv')
print(vic_elec.head())

# DEMAND FOR EACH YEAR
# in R: vic_elec %>% gg_season(Demand)
def quarter():
    vic_elec['quarter'] = pd.PeriodIndex(vic_elec['Date'], freq='Q')
    vic_elec['quarter-str'] = vic_elec['quarter'].values.astype(str)
    vic_elec[['YY','QQ']] = vic_elec['quarter-str'].str.split('Q', expand=True)
    print(vic_elec.head())
    fig2, ax = plt.subplots()
    colors = {'2012':'red', '2013':'green', '2014':'blue'}
    grouped = vic_elec.groupby(['YY'])
    for key, group in grouped:
        group.plot(ax=ax,  x='QQ', y='Demand', label=key, color=colors[key],figsize=(12,6))
    plt.xlabel('QUARTERS')
    plt.ylabel('DEMAND')
    plt.title('DEMAND FOR EACH YEAR')
    plt.show()

# DEMAND FOR EACH 30 MIN.
# IN R: vic_elec %>% gg_season(Demand, period = "day")
vic_elec['Time'] = pd.PeriodIndex(vic_elec['Time'], freq='30T')
print(vic_elec['Time'].head(30))
vic_elec['Time-str'] = vic_elec['Time'].values.astype(str)
vic_elec[['Y','HH']] = vic_elec['Time-str'].str.split(' ', expand=True)

def day():
    plt.figure(figsize=(12,6))
    plt.plot(vic_elec['HH'],vic_elec['Demand'])
    plt.xticks(rotation = 45)
    plt.xlabel('HOURS')
    plt.ylabel('DEMAND')
    plt.title('DEMAND FOR EACH 30 MIN.')
    plt.show()

# scatterplot DEMAND FOR EACH 30 MIN. different colors for each year
def day_scat():
    plt.figure(figsize=(11,6))
    scatter = plt.scatter(vic_elec['HH'], vic_elec['Demand'], c= vic_elec['YY'].astype('category').cat.codes)
    plt.legend(handles=scatter.legend_elements()[0], labels= ['2012','2013','2014'], title="Years:")
    plt.xlabel('HOURS')
    plt.xticks(rotation = 45)
    plt.ylabel('DEMAND')
    plt.title('DEMAND FOR EACH 30 MIN.')
    plt.show()

# DEMAND FOR EACH 30 MIN. different colors for each year
def day_col():
    plt.figure(figsize=(11,6))
    sns.lineplot(vic_elec['HH'], vic_elec['Demand'], hue=vic_elec['Date'], legend='auto')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2)
    plt.xlabel('HOURS')
    plt.xticks(rotation = 45)
    plt.ylabel('DEMAND')
    plt.title('DEMAND FOR EACH 30 MIN.')
    plt.show()



# DEMAND FOR EACH DAY
def week():
    vic_elec['Date2'] = pd.to_datetime(vic_elec['Date'])
    vic_elec['Name_of_the_day'] = vic_elec['Date2'].dt.day_name()
    vic_elec['nameday_HH'] = vic_elec['HH'] + vic_elec['Name_of_the_day']
    print(vic_elec['nameday_HH'])
    plt.figure(figsize=(15,6))
    sns.lineplot(vic_elec['nameday_HH'], vic_elec['Demand'], hue=vic_elec['Date'], legend='auto')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2)
    plt.xticks(rotation = 45)
    plt.xlabel('DAYS')
    plt.ylabel('DEMAND')
    plt.title('DEMAND FOR EACH DAY')
    plt.show()



quarter()
day_scat()
day_col()
day()
week()

