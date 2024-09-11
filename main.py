import pickle
import pandas as pd
import numpy as np
from classmodule import *
import dash
from dash import html, dcc, callback, dash_table
from dash.dependencies import Input, Output, State, ALL
import dash_bootstrap_components as dbc
from dash_bootstrap_templates import load_figure_template
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from datetime import datetime, timedelta

def get_total_usage_per_adress(loaded_data):
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

    dictionary = {'Adres': addresses, 'Gebruik': usage}
    df = pd.DataFrame(dictionary)
    return (df)

def get_daily_usage_per_address(loaded_data):
    addresses = []
    usage = []

    for building in loaded_data.buildings:
        for house in building.house_list:
            address = house.street_name + ' ' + house.house_number  # Full address
            weekly_usage = []  # Store weekly usage per meter

            for meter in house.meters:
                if meter.meter_type == 'WKV':
                    units = meter.units_data

                    if not units.empty:
                        # Convert the index to datetime if it's not already
                        units.index = pd.to_datetime(units.index)

                        # Resample to weekly and safely compute the difference between the last and first reading
                        def safe_diff(x):
                            if len(x) > 1:
                                return x.iloc[-1] - x.iloc[0]
                            else:
                                return 0  # If there is only one or no data points in the week, return 0

                        weekly_data = units.resample('D').apply(safe_diff)

                        # Handle cases where the usage difference is negative or invalid
                        weekly_data[weekly_data < 0] = 0

                        # Append weekly usage
                        weekly_usage.append(weekly_data.values.flatten())

            if weekly_usage:
                # Find the maximum length of weekly usage across all meters
                max_length = max(len(w) for w in weekly_usage)

                # Pad each usage array with zeros to match the maximum length
                padded_usage = [np.pad(w, (0, max_length - len(w)), 'constant') for w in weekly_usage]

                # Sum the padded weekly usage across all meters for the house
                total_weekly_usage = np.sum(padded_usage, axis=0)

                # Append the address and weekly usage
                for day, usage_value in enumerate(total_weekly_usage):
                    addresses.append(f"{address} (Day {day + 1})")
                    usage.append(usage_value)

    # Create a DataFrame for the result
    df = pd.DataFrame({'Adres': addresses, 'Gebruik': usage})

    return df

def make_heatmap(df, filename):

    # Define grid dimensions for both sections
    x_squares_left = 10  # Left section (larger) with 10 columns
    x_squares_right = 4  # Right section (smaller) with 4 columns
    y_squares = 4  # Both sections have 4 rows

    # Fill missing values with zero
    # Calculate the aspect ratio
    aspect_ratio_left = x_squares_left / y_squares
    aspect_ratio_right = x_squares_right / y_squares

    # Create a 2D grid of squares for both sections
    fig = go.Figure()

    # Addresses for the left and right sections of the left building
    addresses_left = [219, 221, 223, 225, 227, 229, 231, 233, 235, 237,
                      217, 215, 213, 211, 209, 207, 205, 203, 201, 199,
                      171, 173, 175, 177, 179, 181, 183, 185, 187, 189,
                      169, 167, 165, 163, 161, 159, 157, 155, 153, 151]
    addresses_right = [191, 193, 195, 197, 149, 147, 145, 143]  # Right side

    # # Addresses for the left and right sections of the right building
    # addresses_left = [77, 79, 81, 83, 85, 87, 89, 91, 93, 95,
    #                         75, 73, 71, 69, 67, 65, 63, 61, 59, 57,
    #                         29, 31, 33, 35, 37, 39, 41, 43, 45, 47,
    #                         27, 25, 23, 21, 19, 17, 15, 13, 11, 9]
    #
    # addresses_right = [49, 51, 53, 55,
    #                          7, 5, 3, 1]

    # Fill NaN values in the 'Gebruik' column with 0 before normalization
    df['Gebruik'] = df['Gebruik'].fillna(0)

    # Normalize "Gebruik" values to range between 0 and 1 for color mapping
    norm_gbruik = (df['Gebruik'] - df['Gebruik'].min()) / (df['Gebruik'].max() - df['Gebruik'].min())

    norm_gbruik = norm_gbruik.fillna(0)

    # Create a color scale that goes from white (255,255,255) to red (255,0,0)
    colors = [f"rgb(255, {int(255 * (1 - r))}, {int(255 * (1 - r))})" for r in norm_gbruik]

    # Plot the left section of the building
    teller = -1
    for j in range(y_squares):  # Loop over rows
        for i in range(x_squares_left):  # Loop over columns for the left section
            teller += 1
            house_number = addresses_left[teller]

            # Match the house number to the corresponding color and "Gebruik" value
            gebruik_value = 0
            if house_number in df['House_Number'].values:
                index = df[df['House_Number'] == house_number].index[0]
                gebruik_value = df.loc[index, 'Gebruik']

            # Force white if the usage value is zero
            if gebruik_value == 0:
                color = "rgb(255,255,255)"  # Set to white for zero usage
            else:
                color = colors[index]  # Use the color from the normalized color list

            # Add square shape for left section
            fig.add_shape(
                type="rect",
                x0=i, x1=i + 1,  # Defines the x-axis position
                y0=y_squares - j - 1, y1=y_squares - j,  # Defines the y-axis position
                fillcolor=color,  # Set color based on "Gebruik"
                line=dict(color="black")
            )

            # Add annotation with house number and "Gebruik" value for left section
            fig.add_annotation(
                x=i + 0.5,  # Center of the square
                y=y_squares - j - 0.3,  # Slightly above the center of the square
                text=f"Verbruik: {gebruik_value}",  # House number
                showarrow=False,
                font=dict(size=15, color="black")
            )

            fig.add_annotation(
                x=i + 0.5,  # Center of the square
                y=y_squares - j - 0.7,  # Slightly below the center of the square
                text=f"{house_number}",  # "Gebruik" value
                showarrow=False,
                font=dict(size=10, color="black")
            )

    # Calculate the correct y-offset to place the right section at ground level
    y_offset = len(addresses_left) // x_squares_left - len(
        addresses_right) // x_squares_right  # Calculate the offset for proper vertical alignment

    # Define the horizontal offset for the right section
    x_offset = 1  # Adjust this value to increase or decrease the gap between the sections

    # Plot the right section of the building
    teller = -1
    for j in range(
            len(addresses_right) // x_squares_right):  # Loop over rows in the right section (should match left section's height)
        for i in range(x_squares_right):  # Loop over columns for the right section
            teller += 1

            # If we exceed the number of addresses in the right section, stop
            if teller >= len(addresses_right):
                break

            house_number = addresses_right[teller]

            # Match the house number to the corresponding color and "Gebruik" value
            gebruik_value = 0
            if house_number in df['House_Number'].values:
                index = df[df['House_Number'] == house_number].index[0]
                gebruik_value = df.loc[index, 'Gebruik']

            # Force white if the usage value is zero
            if gebruik_value == 0:
                color = "rgb(255,255,255)"  # Set to white for zero usage
            else:
                color = colors[index]  # Use the color from the normalized color list

            # Add square shape for right section with adjusted y_offset
            fig.add_shape(
                type="rect",
                x0=x_squares_left + i + x_offset, x1=x_squares_left + i + 1 + x_offset,  # Add x_offset to create space
                y0=y_squares - j - 1 - y_offset, y1=y_squares - j - y_offset,
                # Adjust y-axis position with y_offset to align with left section
                fillcolor=color,  # Set color based on "Gebruik"
                line=dict(color="black")
            )

            # Add annotation with house number and "Gebruik" value for right section
            fig.add_annotation(
                x=x_squares_left + i + 0.5 + x_offset,  # Center of the square in the right section
                y=y_squares - j - 0.3 - y_offset,  # Slightly above the center of the square with y_offset
                text=f"Verbruik: {gebruik_value}",  # House number
                showarrow=False,
                font=dict(size=15, color="black")
            )

            fig.add_annotation(
                x=x_squares_left + i + 0.5 + x_offset,  # Center of the square in the right section
                y=y_squares - j - 0.7 - y_offset,  # Slightly below the center of the square with y_offset
                text=f"{house_number}",  # "Gebruik" value
                showarrow=False,
                font=dict(size=10, color="black")
            )
    filename_without_extension = filename.replace('.jpeg', '')
    # Update the layout to maintain square shapes for both sections
    fig.update_layout(
        autosize=False,  # Manually set the figure size for static images
        width=2600,  # Set width explicitly for the saved image
        height=600,  # Set height explicitly for the saved image
        title=filename_without_extension,  # Add title
        title_x=0.5,  # Center the title
        title_y=0.95,  # Add some space to ensure the title is visible
        font=dict(size=15, color="black"),  # Set font size and color
        xaxis=dict(showgrid=False, zeroline=False, visible=False,
                   range=[0, x_squares_left + x_squares_right + 2]),
        yaxis=dict(showgrid=False, zeroline=False, visible=False, range=[0, y_squares]),
        margin=dict(l=10, r=10, t=80, b=10),  # Add more margin on top for the title
        paper_bgcolor="white",  # Set the background outside the plot to white
        plot_bgcolor="white",  # Set the plot background to white
    )

    fig.write_image(f"C:\\Users\\Thomas Rugers\\Desktop\\Heatmaps\\Left_side\\{filename}", width=2600, height=600)
    return fig


def main():
    with open(
            r"C:\Users\Thomas Rugers\PycharmProjects\Heatmaps\data_Redemptoristenstraat_2024-01-01-2024-09-01.pkl",
            'rb') as file:
        loaded_data = pickle.load(file)

    # Get total and daily usage data
    # total_usage = get_total_usage_per_adress(loaded_data)
    # total_usage['House_Number'] = total_usage['Adres'].str.extract(r'(\d+)').astype(int)
    # total_usage.to_excel(f'C:/Users/Thomas Rugers/Desktop/total_usage.xlsx', index=False, engine='openpyxl')

    daily_usage = get_daily_usage_per_address(loaded_data)
    daily_usage['House_Number'] = daily_usage['Adres'].str.extract(r'(\d+)').astype(int)
    #daily_usage.to_excel(f'C:/Users/Thomas Rugers/Desktop/daily_usage.xlsx', index=False, engine='openpyxl')

    # Extract the day number from the 'Adres' column
    daily_usage['Day'] = daily_usage['Adres'].str.extract(r'\(Day (\d+)\)').astype(int)

    # Pivot the data so that each apartment's daily usage is in a new column
    daily_usage_pivoted = daily_usage.pivot(index='House_Number', columns='Day', values='Gebruik')

    # Set a starting date
    start_date = datetime(2024, 1, 1)

    # Extract individual dataframes for each day and store them in a list
    daily_usage_list = [
        daily_usage_pivoted[[day]].rename(columns={day: 'Gebruik'}).reset_index()
        for day in daily_usage_pivoted.columns
    ]

    # Generate heatmaps for all dataframes and save each with a date in the filename
    for day, df in enumerate(daily_usage_list, start=1):
        # Calculate the date for each day
        current_date = start_date + timedelta(days=day - 1)
        print(day, " ", current_date)
        date_string = current_date.strftime('%Y-%m-%d')  # Format as 'YYYY-MM-DD'
        make_heatmap(df, f"Redemptoristenstraat_143-237_{date_string}.jpeg")

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
