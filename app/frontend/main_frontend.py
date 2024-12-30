import streamlit as st
import requests
from datetime import datetime

backend_url = "https://at3-app-backend-latest.onrender.com"

st.set_page_config(layout='wide')

with st.container() :

    # Headers and Descriptions
    st.header(':rainbow[Plan your next US journey!] :airplane:', divider=True)

    st.markdown("""                        
    Pick your own choice:  
    - :blue[Departure_date / Origin_DepartureTime / Origin_airport / Destination_airport / Cabin Type(s)]
    """)
    
    st.subheader(":blue[Predict your flight fare] :balloon:", divider='grey')


    #Display in three columns 
    left, middle, right = st.columns(3, vertical_alignment="bottom")

    # Input Parameters
    date = left.date_input(':grey[Date]', value=datetime.today())
    time = left.time_input(':grey[Flight Time]')

    all_airports = [
        "ATL", "BOS", "CLT", "DEN", "DFW", "DTW", "EWR", "IAD", "JFK", "LAX",
        "LGA", "MIA", "OAK", "ORD", "PHL", "SFO"
    ]

    startingAirport = middle.selectbox(':grey[Select Origin Departure Airport]', all_airports)

    # Filter destination airports to exclude the selected starting airport
    filtered_airports = [airport for airport in all_airports if airport != startingAirport]
    destinationAirport = middle.selectbox(':grey[Select Destination Airport]', filtered_airports)

    n_stops = right.radio(':grey[Select number of stops]', [0, 1, 2, 3])

    cabin_Leg1 = right.selectbox(':grey[Select Cabin Type 1]', ["coach", "premium coach", "business", "first"])

    # Conditional Cabin Type Selections
    if n_stops == 1:
        cabin_Leg2 = right.selectbox(':grey[Select Cabin Type 2]', ["coach", "premium coach", "business", "first"])
        cabin_Leg3 = cabin_Leg4 = 'no_stop'
    elif n_stops == 2:
        cabin_Leg2 = right.selectbox(':grey[Select Cabin Type 2]', ["coach", "premium coach", "business", "first"])
        cabin_Leg3 = right.selectbox(':grey[Select Cabin Type 3]', ["coach", "premium coach", "business", "first"])
        cabin_Leg4 = 'no_stop'
    elif n_stops == 3:
        cabin_Leg2 = right.selectbox(':grey[Select Cabin Type 2]', ["coach", "premium coach", "business", "first"])
        cabin_Leg3 = right.selectbox(':grey[Select Cabin Type 3]', ["coach", "premium coach", "business", "first"])
        cabin_Leg4 = right.selectbox(':grey[Select Cabin Type 4]', ["coach", "premium coach", "business", "first"])
    else:
        cabin_Leg2 = cabin_Leg3 = cabin_Leg4 = 'no_stop'



# Prediction Button and Request
if st.button("Predict"):
    features = {
        "date": date.strftime('%Y-%m-%d'),
        "time": time.strftime('%H-%M-%S'),  
        "startingAirport": startingAirport,
        "destinationAirport": destinationAirport,
        "n_stops": n_stops,
        "cabin_Leg1": cabin_Leg1,
        "cabin_Leg2": cabin_Leg2,
        "cabin_Leg3": cabin_Leg3,
        "cabin_Leg4": cabin_Leg4
    }

    try:
        # Make a GET request to the backend
        response = requests.get(f'{backend_url}/fare/prediction', params=features)
        if response.status_code == 200:
            output = response.json()

            st.markdown(':blue[Your choice:]:sparkles:')
            st.json(features)
             # Customize the output display
            estimated_fare = output[0]  
            if estimated_fare is not None:
                st.success(f"Your estimated fare is: ${estimated_fare:.2f} , Have a safe trip :sun_with_face:!")
            else:
                st.warning("No fare data available in the response.")

        else:
            st.error("Error fetching data from the backend.")
    except requests.exceptions.RequestException as e:
        st.error(f"Request failed: {e}")