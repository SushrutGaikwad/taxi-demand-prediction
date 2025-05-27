import joblib
import pandas as pd
import streamlit as st
from pathlib import Path
import datetime as dt
from sklearn.pipeline import Pipeline
from sklearn import set_config
from time import sleep


set_config(transform_output="pandas")

# Paths
ROOT_PATH = Path(__file__).parent

PLOT_DATA_PATH = ROOT_PATH / "data" / "external" / "plot_data.csv"
TEST_DATA_PATH = ROOT_PATH / "data" / "processed" / "test.csv"
MODELS_DIR_PATH = ROOT_PATH / "models"

KMEANS_OBJ_PATH = MODELS_DIR_PATH / "mb_kmeans.joblib"
SCALER_OBJ_PATH = MODELS_DIR_PATH / "scaler.joblib"
ENCODER_PATH = MODELS_DIR_PATH / "encoder.joblib"
MODEL_PATH = MODELS_DIR_PATH / "model.joblib"

# Loading the objects
scaler = joblib.load(SCALER_OBJ_PATH)
encoder = joblib.load(ENCODER_PATH)
model = joblib.load(MODEL_PATH)
kmeans = joblib.load(KMEANS_OBJ_PATH)

# Loading the data to plot
plot_df = pd.read_csv(PLOT_DATA_PATH)

# Loading the test data
test_df = pd.read_csv(
    TEST_DATA_PATH,
    parse_dates=["tpep_pickup_datetime"],
)
test_df.set_index("tpep_pickup_datetime", inplace=True)

# UI of app

# Title for the page
st.title("Taxi Demand in New York City")

# Selecting for only neighbors or entire NYC
st.sidebar.title("Options")
map_type = st.sidebar.radio(
    label="Select Map Type:",
    options=["Complete New York City", "Only Neighborhood Regions"],
    index=1,
)

# Header
st.header("Date, Time, & Location")

# Selecting the date
st.subheader("Date")
selected_date = st.date_input(
    "Select the date:",
    value=None,
    min_value=dt.date(year=2016, month=3, day=1),
    max_value=dt.date(year=2016, month=3, day=31),
)
st.write("**Date Selected:**", selected_date)

# Select the time of day
st.subheader("Time")
selected_time = st.time_input(
    "Select the time:",
    value=None,
    step=dt.timedelta(minutes=15),
)
st.write("**Time Selected:**", selected_time)

if selected_date and selected_time:

    # Next time interval
    delta = dt.timedelta(minutes=15)
    current_time = dt.datetime(
        year=selected_date.year,
        month=selected_date.month,
        day=selected_date.day,
        hour=selected_time.hour,
        minute=selected_time.minute,
    )
    next_time_interval = current_time + delta
    st.write(f"Demand forecast requested at time: {next_time_interval.time()}")

    # Creating the datetime index
    current_time_index = pd.Timestamp(f"{selected_date} {current_time.time()}")
    next_time_interval_index = pd.Timestamp(f"{selected_date} {next_time_interval.time()}")
    st.write("**Date & Time of Forecasted Demand:**", next_time_interval_index)

    # Random sampling a latitude longitude pair
    st.subheader("Location")
    sample_input = plot_df.sample(1).reset_index(drop=True)
    latitude = sample_input["pickup_latitude"].item()
    longitude = sample_input["pickup_longitude"].item()
    region = sample_input["region"].item()
    st.write("**Your Current Location and Region:**")
    st.write(f"(Latitude, Longitude): ({latitude}, {longitude})")

    with st.spinner("Fetching your current region..."):
        sleep(3)

    st.write(f"Region ID: {region}")
    # Scaling the random sample
    scaled_sample_input = scaler.transform(sample_input.iloc[:, 0:2])

    # Plotting the map
    st.header("Map")

    # List of 30 hex colors
    colors = [
        "#FF0000",
        "#FF4500",
        "#FF8C00",
        "#FFD700",
        "#ADFF2F",
        "#32CD32",
        "#008000",
        "#006400",
        "#00FF00",
        "#7CFC00",
        "#00FA9A",
        "#00FFFF",
        "#40E0D0",
        "#4682B4",
        "#1E90FF",
        "#0000FF",
        "#0000CD",
        "#8A2BE2",
        "#9932CC",
        "#BA55D3",
        "#FF00FF",
        "#FF1493",
        "#C71585",
        "#FF4500",
        "#FF6347",
        "#FFA07A",
        "#FFDAB9",
        "#FFE4B5",
        "#F5DEB3",
        "#EEE8AA",
    ]

    # Adding color to the data
    regions = plot_df["region"].unique().tolist()
    region_colors = {region: colors[i] for i, region in enumerate(regions)}
    plot_df["color"] = plot_df["region"].map(region_colors)

    # Prediction pipeline
    pipe = Pipeline(
        [
            ("encoder", encoder),
            ("model", model),
        ]
    )

    if map_type == "Complete New York City":
        # Progress bar
        progress_bar = st.progress(value=0, text="Operation in progress. Please wait...")
        for percent_complete in range(100):
            sleep(0.05)
            progress_bar.progress(
                percent_complete + 1, text="Operation in progress. Please wait..."
            )

        # Map
        st.map(
            data=plot_df,
            latitude="pickup_latitude",
            longitude="pickup_longitude",
            size=0.01,
            color="color",
        )

        # Removing the progress bar
        progress_bar.empty()

        # Filtering the data for all 30 regions
        input_data = test_df.loc[current_time_index, :].sort_values("region")

        # Doing the prediction for all 30 regions
        predictions = pipe.predict(input_data.drop(columns=["total_pickups"]))

        # Showing the map labels

        # Displaying the map legend
        st.subheader("Map Legend")
        for ind in range(0, 30):
            color = colors[ind]
            demand = predictions[ind]
            if region == ind:
                region_id = f"{ind} (Current region)"
            else:
                region_id = ind
            st.markdown(
                f'<div style="display: flex; align-items: center;">'
                f'<div style="background-color:{color}; width: 20px; height: 10px; margin-right: 10px;"></div>'
                f"Region ID: {region_id} <br>"
                f"Demand: {int(demand)} <br> <br>",
                unsafe_allow_html=True,
            )

    else:

        # Calculating the distances from centroid
        distances = kmeans.transform(scaled_sample_input.values).values.ravel().tolist()
        distances = list(enumerate(distances))
        sorted_distances = sorted(distances, key=lambda x: x[1])[0:9]

        indexes_of_closest_9_regions = []
        for ind in sorted_distances:
            indexes_of_closest_9_regions.append(ind[0])

        # Filtering plot data and considering only the closest 9 regions
        plot_df_filtered = plot_df[plot_df["region"].isin(indexes_of_closest_9_regions)]

        # Progress bar
        progress_bar = st.progress(value=0, text="Operation in progress. Please wait...")
        for percent_complete in range(100):
            sleep(0.05)
            progress_bar.progress(
                percent_complete + 1, text="Operation in progress. Please wait..."
            )

        # Map
        st.map(
            data=plot_df_filtered,
            latitude="pickup_latitude",
            longitude="pickup_longitude",
            size=0.01,
            color="color",
        )

        # Removing the progress bar
        progress_bar.empty()

        # Filtering the data for only the closest 9 regions
        input_data = test_df.loc[current_time_index, :]
        input_data = input_data.loc[
            input_data["region"].isin(indexes_of_closest_9_regions), :
        ].sort_values("region")

        # Doing the predictions for only the closest 9 regions
        predictions = pipe.predict(input_data.drop(columns=["total_pickups"]))

        # Showing the map labels

        # Displaying the map legend
        st.subheader("Map Legend")
        for ind in range(0, 9):
            color = colors[indexes_of_closest_9_regions[ind]]
            demand = predictions[ind]
            if region == indexes_of_closest_9_regions[ind]:
                region_id = f"{indexes_of_closest_9_regions[ind]} (Current region)"
            else:
                region_id = indexes_of_closest_9_regions[ind]
            st.markdown(
                f'<div style="display: flex; align-items: center;">'
                f'<div style="background-color:{color}; width: 20px; height: 10px; margin-right: 10px;"></div>'
                f"Region ID: {region_id} <br>"
                f"Demand: {int(demand)} <br> <br>",
                unsafe_allow_html=True,
            )
