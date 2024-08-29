# This is a sample Python script.
import pickle
import pandas as pd
import numpy as np
from classmodule import *

# Press the green button in the gutter to run the script.
def main():
    with open(r"C:\Users\Thomas Rugers\PycharmProjects\Tvd_test\data_Kruidenlaan_2024-01-01-2024-01-05.pkl", 'rb') as file:
        loaded_data = pickle.load(file)
    print("Het metertype = " + loaded_data.buildings[0].house_list[0].meters[0].meter_type)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
     main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
