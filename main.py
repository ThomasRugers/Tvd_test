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

def scatterHeatmap(df):
    # Define grid dimensions for both sections
    x_squares_left = 10  # Left section (larger) with 10 columns
    x_squares_right = 4  # Right section (smaller) with 4 columns
    y_squares = 4  # Both sections have 4 rows

    # Calculate the aspect ratio
    aspect_ratio_left = x_squares_left / y_squares
    aspect_ratio_right = x_squares_right / y_squares

    # Create a 2D grid of squares for both sections
    fig = go.Figure()

    # Addresses for the left and right sections of the building
    addresses_left = [219, 221, 223, 225, 227, 229, 231, 233, 235, 237,
                      217, 215, 213, 211, 209, 207, 205, 203, 201, 199,
                      171, 173, 175, 177, 179, 181, 183, 185, 187, 189,
                      169, 167, 165, 163, 161, 159, 157, 155, 153, 151]
    addresses_right = [191, 193, 195, 197, 149, 147, 145, 143]  # Right side

    # Normalize "Gebruik" values to range between 0 and 1 for color mapping
    norm_gbruik = (df['Gebruik'] - df['Gebruik'].min()) / (df['Gebruik'].max() - df['Gebruik'].min())

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

    # Update the layout to maintain square shapes for both sections
    fig.update_layout(
        autosize=False,  # Manually set the figure size for static images
        width=2600,  # Set width explicitly for the saved image
        height=600,  # Set height explicitly for the saved image
        title='Redemptoristenstraat, Adressen 143-237, 01-01-2024 tot 09-01-2024',  # Add title
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

    fig.write_image("C:\\Users\\Thomas Rugers\\Desktop\\scatter_heatmap.jpeg", width=2600, height=600)

    # Initialize Dash app
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

    # App layout
    app.layout = dbc.Container([
        dbc.NavbarSimple(
            brand="Grid Visualization",
            brand_href="#",
            color="primary",
            dark=True,
        ),
        dbc.Row([
            dbc.Col([
                html.H2("Verbruik per adres"),
                dcc.Graph(
                    id='grid',
                    figure=fig
                )
            ], width=12)
        ]),
    ], fluid=True)

    # Run the Dash server here (do not add __main__ check inside this function)
    app.run_server(debug=True, port=8051)
    return fig

def create_heatmap_for_day_fixed_v2(day, df_day):
    # Define your left and right addresses sections
    addresses_left = [219, 221, 223, 225, 227, 229, 231, 233, 235, 237,
                      217, 215, 213, 211, 209, 207, 205, 203, 201, 199,
                      171, 173, 175, 177, 179, 181, 183, 185, 187, 189,
                      169, 167, 165, 163, 161, 159, 157, 155, 153, 151]
    addresses_right = [191, 193, 195, 197, 149, 147, 145, 143]

    # Create a 2D grid of squares for the day
    fig = go.Figure()

    # Plot the left section of the building
    teller = -1
    for j in range(4):  # Loop over rows
        for i in range(10):  # Loop over columns for the left section
            teller += 1
            if teller >= len(addresses_left):
                break
            house_number = addresses_left[teller]

            # Check if the house number exists in the filtered dataframe for this day
            if house_number in df_day['House_Number'].values:
                index = df_day[df_day['House_Number'] == house_number].index[0]
                gebruik_value = df_day.loc[index, 'Gebruik']
                # Normalize the value to set the color
                norm_value = (gebruik_value - df_day['Gebruik'].min()) / (df_day['Gebruik'].max() - df_day['Gebruik'].min())
                color = f"rgb(255, {int(255 * (1 - norm_value))}, {int(255 * (1 - norm_value))})"
            else:
                gebruik_value = 0
                color = "rgb(255,255,255)"  # Force white for house numbers not present

            # Add square shape for left section
            fig.add_shape(
                type="rect",
                x0=i, x1=i + 1,  # Defines the x-axis position
                y0=4 - j - 1, y1=4 - j,  # Defines the y-axis position
                fillcolor=color,  # Set color based on "Gebruik"
                line=dict(color="black")
            )

            # Add annotation with house number and "Gebruik" value for left section
            fig.add_annotation(
                x=i + 0.5,  # Center of the square
                y=4 - j - 0.3,  # Slightly above the center of the square
                text=f"Verbruik: {gebruik_value}",  # House number
                showarrow=False,
                font=dict(size=15, color="black")
            )

            fig.add_annotation(
                x=i + 0.5,  # Center of the square
                y=4 - j - 0.7,  # Slightly below the center of the square
                text=f"{house_number}",  # "Gebruik" value
                showarrow=False,
                font=dict(size=10, color="black")
            )

    # Plot the right section of the building
    teller = -1
    for j in range(2):  # Loop over rows in the right section
        for i in range(4):  # Loop over columns for the right section
            teller += 1
            if teller >= len(addresses_right):
                break
            house_number = addresses_right[teller]

            # Check if the house number exists in the filtered dataframe for this day
            if house_number in df_day['House_Number'].values:
                index = df_day[df_day['House_Number'] == house_number].index[0]
                gebruik_value = df_day.loc[index, 'Gebruik']
                # Normalize the value to set the color
                norm_value = (gebruik_value - df_day['Gebruik'].min()) / (df_day['Gebruik'].max() - df_day['Gebruik'].min())
                color = f"rgb(255, {int(255 * (1 - norm_value))}, {int(255 * (1 - norm_value))})"
            else:
                gebruik_value = 0
                color = "rgb(255,255,255)"  # Force white for house numbers not present

            # Add square shape for right section
            fig.add_shape(
                type="rect",
                x0=10 + i, x1=10 + i + 1,  # Add x_offset to create space
                y0=4 - j - 1, y1=4 - j,
                fillcolor=color,  # Set color based on "Gebruik"
                line=dict(color="black")
            )

            # Add annotation with house number and "Gebruik" value for right section
            fig.add_annotation(
                x=10 + i + 0.5,  # Center of the square in the right section
                y=4 - j - 0.3,  # Slightly above the center of the square
                text=f"Verbruik: {gebruik_value}",  # House number
                showarrow=False,
                font=dict(size=15, color="black")
            )

            fig.add_annotation(
                x=10 + i + 0.5,  # Center of the square in the right section
                y=4 - j - 0.7,  # Slightly below the center of the square
                text=f"{house_number}",  # "Gebruik" value
                showarrow=False,
                font=dict(size=10, color="black")
            )

    # Update the layout to maintain square shapes
    fig.update_layout(
        autosize=False,  # Manually set the figure size for static images
        width=2400,  # Set width explicitly for the saved image
        height=600,  # Set height explicitly for the saved image
        title=f'Heatmap for {day}',  # Add title
        title_x=0.5,  # Center the title
        title_y=0.95,  # Add some space to ensure the title is visible
        font=dict(size=15, color="black"),  # Set font size and color
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False),
        margin=dict(l=10, r=10, t=80, b=10),  # Add more margin on top for the title
        paper_bgcolor="white",  # Set the background outside the plot to white
        plot_bgcolor="white",  # Set the plot background to white
    )

    # Save the figure as a JPEG image
    output_path = f'C:/Users/Thomas Rugers/Desktop/Heatmaps/heatmap_{day}.jpeg'
    fig.write_image(output_path, width=1600, height=600)
    return output_path

def get_weekly_usage_per_address(loaded_data):
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

def create_overview_df(loaded_data):
    # Lijsten om de informatie op te slaan
    building_names = []
    house_addresses = []
    meter_types = []
    meter_readings = []
    reading_timestamps = []

    # Loop door de gebouwen in de loaded_data
    for building in loaded_data.buildings:
        for house in building.house_list:
            address = house.street_name + ' ' + house.house_number  # Volledig adres
            for meter in house.meters:
                if meter.meter_type == 'WKV':  # Alleen WKV meters toevoegen
                    # Koppel de meterstanden aan de tijdstempels
                    units = meter.units_data
                    if not units.empty:
                        # Gebruik iterrows() om over de DataFrame te itereren
                        for timestamp, row in units.iterrows():
                            # Voeg de gegevens toe aan de respectieve lijsten
                            building_names.append(building.name)
                            house_addresses.append(address)
                            meter_types.append(meter.meter_type)
                            meter_readings.append(row.values[0])  # De waarde van de meterstand
                            reading_timestamps.append(timestamp)

    # Maak een dictionary met de verzamelde data
    data = {
        'Gebouw': building_names,
        'Adres': house_addresses,
        'Meter Type': meter_types,
        'Meterstand': meter_readings,
        'Tijdstempel': reading_timestamps
    }

    # Maak een DataFrame van het overzicht
    df_overview = pd.DataFrame(data)

    return df_overview

# Main entry point for the script
def main():
    # Load the data
    with open(r"C:\Users\Thomas Rugers\PycharmProjects\Tvd_test\data_Redemptoristenstraat_2024-01-01-2024-09-01.pkl", 'rb') as file:
        loaded_data = pickle.load(file)

    # Create the df with the total usage per address
    df = get_total_usage_per_adress(loaded_data)
    df2 = get_weekly_usage_per_address(loaded_data)

    # Extract the 'Day' column from 'Adres'
    df2['Day'] = df2['Adres'].str.extract(r'\(Day (\d+)\)')

    # Extract the house number from 'Adres'
    df2['House_Number'] = df2['Adres'].str.extract(r'(\d+)').astype(int)

    # Loop through unique days and generate heatmaps
    heatmap_files_fixed_v2 = []
    for day in df2['Day'].unique():
        df_day = df2[df2['Day'] == day]
        heatmap_files_fixed_v2.append(create_heatmap_for_day_fixed_v2(day, df_day))

    # Create the scatter heatmap
    scatterHeatmap(df)


if __name__ == '__main__':
    main()
