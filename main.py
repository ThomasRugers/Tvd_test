# This is a sample Python script.
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
                text=f"Gebruik: {gebruik_value}",  # House number
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
                text=f"Gebruik: {gebruik_value}",  # House number
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
        title='Redemptoristenstraat 01-01-2024 tot 09-01-2024',  # Add title
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
                html.H2("Gebruik per adres"),
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



# Main entry point for the script
def main():
    with open(r"C:\Users\Thomas Rugers\PycharmProjects\Tvd_test\data_Redemptoristenstraat_2024-01-01-2024-09-01.pkl", 'rb') as file:
        loaded_data = pickle.load(file)
    # Create the df with the total usage per address
    df = get_total_usage_per_adress(loaded_data)

    # Create the scatter heatmap
    df['House_Number'] = df['Adres'].str.extract('(\d+)').astype(int)
    scatterHeatmap(df)



if __name__ == '__main__':
    main()
