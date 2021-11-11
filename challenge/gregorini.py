import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('electricity.csv')

def yearly_graph():
    data['my_dates'] = pd.to_datetime(data['Date'])
    data["Days"] = data["Time"].str.split('-')
    close_list = data["Days"].tolist()
    lista = list(zip(*close_list))
    listaa = list(lista[0])
    data["Day"]=listaa
    data["Day"] = data["Time"].str.split('-')
    close_list = data["Day"].tolist()
    lista = list(zip(*close_list))
    listaa = list(lista[1])
    listaa = [int(x) for x in listaa]
    data["Day"] = listaa

    data["Dayss"] = data["Time"].str.split('-')
    data["Dayss"] = data["Time"].str.split('T')
    close_list1 = data["Dayss"].tolist()
    listaw = list(zip(*close_list1))
    listaaa = list(listaw[0])
    data["Dayss"] = listaaa

    data["Dayss"] = data["Dayss"].str.split('-',1)
    close_list1 = data["Dayss"].tolist()
    listaw = list(zip(*close_list1))
    listaaa = list(listaw[1])
    data["Dayss"] = listaaa

    listaaa1 = list(listaw[0])
    listaaa1 = [int(x) for x in listaaa1]
    data["Year"] = listaaa1

    for i in range(2011, 2015):
        x_i = (data[data["Year"] == i])
        y_i = (data[data["Year"] == i])
        plt.plot(x_i["Dayss"], y_i["Demand"])

    plt.xlabel('First Quarter  Second Quarter  Third Quarter  Fourth Quarter', loc = "center")
    plt.title('Electricity consumption by quarter')
    plt.legend(loc="upper right")
    plt.ylabel('Energy consumption')
    plt.show()


def daily_graph():
    data["Days"] = data["Time"].str.split('T')
    close_list = data["Days"].tolist()
    lista = list(zip(*close_list))
    listaa = list(lista[1])
    listaa = [x[:-1] for x in listaa]
    data["Day"]=listaa
    sns.lineplot(data["Day"], data["Demand"], hue = data['Date'])
    plt.title('Electricity consumption by half-hours')
    plt.xticks(rotation = 60)
    plt.xlabel('Hour')
    plt.ylabel('Energy consumption')
    plt.show()

def weekly_graph():
    data['my_dates'] = pd.to_datetime(data['Date'])
    data['day_of_week'] = data['my_dates'].dt.day_name()
    data["Days"] = data["Time"].str.split('T')
    close_list = data["Days"].tolist()
    lista = list(zip(*close_list))
    listaa = list(lista[1])
    listaa = [x[:-1] for x in listaa]
    data["Day"] = listaa
    data['hour'] = pd.to_datetime(data['Day'])
    data["graph"] = data['Day'] + data['day_of_week']
    sns.lineplot(data["graph"], data["Demand"], hue = data['Date'], legend = "auto")
    plt.title('Electricity consumption by weekday')
    plt.xlabel('Monday Tuesday Wednesday Thursday Friday Saturday Sunday', loc = "center")
    plt.ylabel('Energy consumption')
    plt.show()

daily_graph()
weekly_graph()
yearly_graph()
