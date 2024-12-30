from fastapi import FastAPI, HTTPException
from starlette.responses import JSONResponse
from joblib import load
import pandas as pd
from datetime import datetime
import numpy as np
from typing import Dict
from enum import Enum
import uvicorn
from catboost import CatBoostRegressor, Pool
np.float_ = np.float64

# Initialize FastAPI app
app = FastAPI()

median_traveldistance = pd.read_csv('models/median_traveldistance.csv')

# Load Model
farhan_model = load('models/pred_pipe_lgb16.joblib')

ratana_model = load('models/xgb_pipeline.joblib')

sharon_model = load('models/xgb.pkl')

# Load the trained CatBoost model (Sakib's model)
model = CatBoostRegressor()
model.load_model('models/catboost_final_model.cbm')

# Load the list of categorical features
categorical_features = load('models/categorical_features.joblib')

# Load the feature columns
feature_columns = load('models/feature_columns.joblib')



@app.get("/")
def read_root():
    return {
        "Project Objectives": (
            "This project focuses on building a machine learning model that predicts airline fare."
        )
    }

@app.get('/health', status_code=200)
def healthcheck():
    return 'Happy predicting!'


#Match median travel distance
def get_distance(median_traveldistance, start, destination):
    df = median_traveldistance[(median_traveldistance['startingAirport']==start) &
                               (median_traveldistance['destinationAirport']==destination)]

    if df is not None:
        distance = df['medianTravelDistance'].values[0]
        return distance
    return None

### RATANA PROCESS FLIGHT DATA ###
def ratana_process_flight_data(
    date: str,
    time: str,
    startingAirport: str,
    destinationAirport: str,
    n_stops: int,
    Cabin_Leg1: str,
    Cabin_Leg2: str = "no_stop", # Set default parameter for no stop
    Cabin_Leg3: str = "no_stop", # Set default parameter for no stop
    Cabin_Leg4: str = "no_stop" # Set default parameter for no stop
):
    """
    Function to process flight data to inputs for modelling
    """
    
    # Check and adjust cabin types based on the number of stops
    if n_stops == 0:
        Cabin_Leg2 = Cabin_Leg3 = Cabin_Leg4 = "no_stop"
    elif n_stops == 1:
        Cabin_Leg3 = Cabin_Leg4 = "no_stop"
    elif n_stops == 2:
        Cabin_Leg4 = "no_stop"
    
    # Extract date and time features (simplified here)
    combined_datetime = datetime.strptime(f"{date} {time}", "%Y-%m-%d %H-%M-%S")

    # Use the current date as the reference flight date
    reference_flight_date = datetime.now()

    # Calculate the date_diff
    date_diff = (combined_datetime - reference_flight_date).days

    # Get the travel distance based on airport combinations
    travel_distance = get_distance(median_traveldistance, startingAirport, destinationAirport)

    # Create the feature dictionary
    features = {
        "month": combined_datetime.month,
        "day": combined_datetime.day,
        "hour": combined_datetime.hour,
        "minute": combined_datetime.minute,
        "day_of_week": combined_datetime.weekday(),  # e.g., Monday
        "week_of_year": combined_datetime.isocalendar()[1],  # ISO week number
        "date_diff": date_diff,
        "startingAirport": startingAirport,
        "destinationAirport": destinationAirport,
        "n_stops": n_stops,
        "medianTravelDistance": travel_distance,
        "Cabin_Leg1": Cabin_Leg1,
        "Cabin_Leg2": Cabin_Leg2,
        "Cabin_Leg3": Cabin_Leg3,
        "Cabin_Leg4": Cabin_Leg4
    }
    # Convert the features dictionary into a Dataframe
    features_df = pd.DataFrame([features])

    # Specify the desired data types
    features_df = features_df.astype({
        "month": "int64",
        "day": "int64",
        "hour": "int64",
        "minute": "int64",
        "day_of_week": "int64",
        "week_of_year": "int64",
        "date_diff": "int64",
        "startingAirport": "string",
        "destinationAirport": "string",
        "n_stops": "int64",
        "medianTravelDistance": "float64",
        "Cabin_Leg1": "string",
        "Cabin_Leg2": "string",
        "Cabin_Leg3": "string",
        "Cabin_Leg4": "string"
        
    })
    return features_df

### FARHAN PROCESS_FLIGHT_DATA ###
def farhan_process_flight_data(
    date: str,
    time: str,
    startingAirport: str,
    destinationAirport: str,
    n_stops: int,
    cabin_Leg1: str,
    cabin_Leg2: str = "no_stop",
    cabin_Leg3: str = "no_stop",
    cabin_Leg4: str = "no_stop"
):
    # Adjust cabin types based on the number of stops
    if n_stops == 0:
        cabin_Leg2 = cabin_Leg3 = cabin_Leg4 = "no_stop"
    elif n_stops == 1:
        cabin_Leg3 = cabin_Leg4 = "no_stop"
    elif n_stops == 2:
        cabin_Leg4 = "no_stop"
    
    # Extract date and time features
    combined_datetime = datetime.strptime(f"{date} {time}", "%Y-%m-%d %H-%M-%S")
    reference_flight_date = datetime.now()
    date_diff = (combined_datetime - reference_flight_date).days

    # Get the travel distance based on airport combinations
    travel_distance = get_distance(median_traveldistance, startingAirport, destinationAirport)

    # Create the feature dictionary
    features = {
        "month": combined_datetime.month,
        "day": combined_datetime.day,
        "hour": combined_datetime.hour,
        "minute": combined_datetime.minute,
        "day_of_week": combined_datetime.weekday(),
        "week_of_year": combined_datetime.isocalendar()[1],
        "date_diff": date_diff,
        "startingAirport": startingAirport,
        "destinationAirport": destinationAirport,
        "n_stops": n_stops,
        "cabin_Leg1": cabin_Leg1,
        "cabin_Leg2": cabin_Leg2,
        "cabin_Leg3": cabin_Leg3,
        "cabin_Leg4": cabin_Leg4,
        "Travel_distance": travel_distance
    }

    # Convert features to a DataFrame
    features_df = pd.DataFrame([features])

    # Specify the data types
    features_df = features_df.astype({
        "month": "int64",
        "day": "int64",
        "hour": "int64",
        "minute": "int64",
        "day_of_week": "int64",
        "week_of_year": "int64",
        "date_diff": "int64",
        "startingAirport": "string",
        "destinationAirport": "string",
        "n_stops": "int64",
        "cabin_Leg1": "string",
        "cabin_Leg2": "string",
        "cabin_Leg3": "string",
        "cabin_Leg4": "string",
        "Travel_distance": "float64"
    })
    return features_df

### SAKIB PROCESS FLIGHT DATA ###

def get_travel_info(startingAirport: str, destinationAirport: str) -> dict:
    """
    Retrieve the median travel distance and duration for given airports.
    """
    result = median_traveldistance[
        (median_traveldistance['startingAirport'] == startingAirport) &
        (median_traveldistance['destinationAirport'] == destinationAirport)
    ]
    if not result.empty:
        travel_distance = float(result['medianTravelDistance'].iloc[0])
        travel_duration = float(result['medianTravelDuration'].iloc[0])
    else:
        # Handle missing data by setting default values or raising an error
        travel_distance = 0.0
        travel_duration = 0.0
    return {'travelDistance': travel_distance, 'travelDuration': travel_duration}



def sakib_process_flight_data(
    date: str,
    time: str,
    startingAirport: str,
    destinationAirport: str,
    n_stops: int,
    cabin_Leg1: str,
    cabin_Leg2: str,
    cabin_Leg3: str,
    cabin_Leg4: str
):
    # Adjust cabin types based on the number of stops
    n_stops = int(n_stops)  # Ensure n_stops is an integer

    if n_stops == 0:
        cabin_Leg2 = cabin_Leg3 = cabin_Leg4 = "no_stop"
    elif n_stops == 1:
        cabin_Leg3 = cabin_Leg4 = "no_stop"
    elif n_stops == 2:
        cabin_Leg4 = "no_stop"
    elif n_stops > 3:
        raise ValueError("Number of stops cannot exceed 3.")

    # Parse date and time
    try:
        combined_datetime = datetime.strptime(f"{date} {time}", "%Y-%m-%d %H-%M-%S")
    except ValueError:
        raise ValueError("Invalid date or time format.")

    current_datetime = datetime.now()
    date_diff = (combined_datetime - current_datetime).days

    # Extract date and time features
    month = combined_datetime.month
    day = combined_datetime.day
    hour = combined_datetime.hour
    minute = combined_datetime.minute
    day_of_week = combined_datetime.weekday()  # Monday is 0
    week_of_year = combined_datetime.isocalendar()[1]

    # Set isNonStop based on n_stops (int64)
    isNonStop = 1 if n_stops == 0 else 0

    # Convert cabin types to lowercase
    cabin_Leg1 = cabin_Leg1.lower()
    cabin_Leg2 = cabin_Leg2.lower()
    cabin_Leg3 = cabin_Leg3.lower()
    cabin_Leg4 = cabin_Leg4.lower()

    # Determine isBasicEconomy based on cabin_Leg1 (int64)
    isBasicEconomy = 1 if cabin_Leg1 == 'coach' else 0

    # Assume default value for isRefundable (int64)
    isRefundable = 0  # Assuming non-refundable tickets by default

    # Get travel distance and duration
    travel_info = get_travel_info(startingAirport, destinationAirport)
    totalTravelDistance = travel_info['travelDistance']
    travelDuration = travel_info['travelDuration']

    if totalTravelDistance == 0.0 or travelDuration == 0.0:
        raise ValueError("Travel information not available for the selected airports.")

    # Create the feature dictionary
    features = {
        "startingAirport": startingAirport,
        "destinationAirport": destinationAirport,
        "travelDuration": travelDuration,
        "isBasicEconomy": isBasicEconomy,
        "isRefundable": isRefundable,
        "isNonStop": isNonStop,
        "totalTravelDistance": totalTravelDistance,
        "month": month,
        "day": day,
        "day_of_week": day_of_week,
        "week_of_year": week_of_year,
        "date_diff": date_diff,
        "hour": hour,
        "minute": minute,
        "cabin_Leg1": cabin_Leg1,
        "cabin_Leg2": cabin_Leg2,
        "cabin_Leg3": cabin_Leg3,
        "cabin_Leg4": cabin_Leg4
    }

    # Convert features to a DataFrame
    features_df = pd.DataFrame([features])

    # Reorder columns to match the training data
    features_df = features_df[feature_columns]

    # Ensure correct data types
    # Set categorical features as strings
    for col in categorical_features:
        features_df[col] = features_df[col].astype('str')

    # Set integer features
    int_features = [
        'isBasicEconomy', 'isRefundable', 'isNonStop', 'month', 'day',
        'day_of_week', 'week_of_year', 'date_diff', 'hour', 'minute'
    ]
    float_features = ['travelDuration', 'totalTravelDistance']

    for col in int_features:
        features_df[col] = features_df[col].astype('int64')

    for col in float_features:
        features_df[col] = features_df[col].astype('float64')

    return features_df


### SHARON PROCESS FLGHT DATA ###

def xgb_format_features(
    date: str, time: str, startingAirport: str, destinationAirport: str,
    n_stops: int, cabin_Leg1: str, cabin_Leg2: str, cabin_Leg3: str, cabin_Leg4: str
):
    date_toconvert = pd.to_datetime(date, format="%Y-%m-%d")
    day = date_toconvert.day
    day_of_week = date_toconvert.dayofweek
    week_of_year = date_toconvert.isocalendar().week
    Month = date_toconvert.month
    searchdate = datetime.today()
    date_diff = (date_toconvert - searchdate).days
    isWeekend = day_of_week in [5, 6]
    time_toconvert = pd.to_datetime(time, format="%H-%M-%S")
    Hour = time_toconvert.hour
    Minute = time_toconvert.minute
    totalTravelDistance = get_distance(median_traveldistance, startingAirport, destinationAirport)

    if n_stops == 0:
        cabin_Leg2 = cabin_Leg3 = cabin_Leg4 = 'no_stop'
    elif n_stops == 1:
        cabin_Leg3 = cabin_Leg4 = 'no_stop'
    elif n_stops == 2:
        cabin_Leg4 = 'no_stop'

    features = {
        'day': day,
        'day_of_week': day_of_week,
        'week_of_year': week_of_year,
        'month': Month,
        'hour': Hour,
        'minute': Minute,
        'date_diff': date_diff,
        'startingAirport': startingAirport,
        'destinationAirport': destinationAirport,
        'n_stops': n_stops,
        'totalTravelDistance': totalTravelDistance,
        'isWeekend': isWeekend,
        'Cabin_Leg1': cabin_Leg1,  
        'Cabin_Leg2': cabin_Leg2,  
        'Cabin_Leg3': cabin_Leg3,  
        'Cabin_Leg4': cabin_Leg4   
    }

    # Convert to DataFrame for model input
    return pd.DataFrame([features])


@app.get("/fare/prediction")
def predict(
    date: str,
    time: str,
    startingAirport: str,
    destinationAirport: str,
    n_stops: int,
    cabin_Leg1: str,
    cabin_Leg2: str = "no_stop",
    cabin_Leg3: str = "no_stop",
    cabin_Leg4: str = "no_stop"
):
     # List of valid airports
    farhan_airports = ['OAK', 'ORD', 'PHL', 'SFO']
    ratana_airports = ['ATL', 'BOS', 'CLT', 'DEN']
    sharon_airports = ['JFK', 'LAX', 'LGA', 'MIA']
    sakib_airports = ['DFW', 'DTW', 'EWR', 'IAD']
    
    # Choose model based on the starting airport
    try:
        if startingAirport in farhan_airports:  # FARHAN SECTION
            obs = farhan_process_flight_data(date, time, startingAirport, destinationAirport, n_stops,
                                             cabin_Leg1, cabin_Leg2, cabin_Leg3, cabin_Leg4)
            pred = farhan_model.predict(obs)

        elif startingAirport in ratana_airports:  # RATANA SECTION
            obs = ratana_process_flight_data(date, time, startingAirport, destinationAirport, n_stops,
                                             cabin_Leg1, cabin_Leg2, cabin_Leg3, cabin_Leg4)
            
            pred = ratana_model.predict(obs)

        elif startingAirport in sakib_airports:  # SAKIB SECTION
            obs = sakib_process_flight_data(date, time, startingAirport, destinationAirport, n_stops,
            cabin_Leg1, cabin_Leg2, cabin_Leg3, cabin_Leg4)

            # Prepare Pool for prediction
            obs_pool = Pool(obs, cat_features=categorical_features)

            # Predict using the loaded model
            pred = model.predict(obs_pool)

        elif startingAirport in sharon_airports:  # SHARON SECTION

            # Prepare features for the Sharon model
            obs = xgb_format_features(
                date, time, startingAirport, destinationAirport,
                n_stops, cabin_Leg1, cabin_Leg2, cabin_Leg3, cabin_Leg4)
            pred = sharon_model.predict(obs)

        else:
            raise HTTPException(status_code=400, detail="Starting airport not supported")

        return JSONResponse(pred.tolist())
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")



