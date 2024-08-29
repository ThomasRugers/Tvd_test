# This is a sample Python script.
import pickle
import pandas as pd
import numpy as np
from classmodule import *

def main():
    with open(r"C:\Users\Thomas Rugers\PycharmProjects\Tvd_test\data_Kruidenlaan_2024-01-01-2024-01-05.pkl", 'rb') as file:
        loaded_data = pickle.load(file)
    print("Het metertype = " + loaded_data.buildings[0].house_list[0].meters[0].meter_type)

    addresses = []
    usage = []

    for building in loaded_data.buildings:
        for house in building.house_list:
            total_usage = 0
            addresses.append(house.street_name + ' ' + house.house_number)
            for meter in house.meters:
                if meter.meter_type == 'WKV':
                    units = meter.units_data
                    total_usage += (units.iloc[-1] - units.iloc[0]).values[0] if not units.empty else 0

            usage.append(total_usage)

    df = pd.DataFrame({'Adres': addresses, 'Gebruik': usage})
    print(df)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
     main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
