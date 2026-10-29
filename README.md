# Flight Fare Prediction App

![Python](https://img.shields.io/badge/python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688)
![Streamlit](https://img.shields.io/badge/Streamlit-1.39-FF4B4B)
![Docker](https://img.shields.io/badge/Docker-ready-2496ED)

A web app that predicts the fare of a US domestic flight. The user picks a date, time, route, number of stops and cabin class for each leg. A FastAPI backend sends the request to one of four trained gradient-boosting models and returns a fare estimate.

**[Live demo](https://at3-app-frontend-latest.onrender.com/)** · **[Research repo and models](https://github.com/RatanaSovann/flight_fare_prediction)** · **[Project report](https://github.com/RatanaSovann/flight_fare_prediction/blob/main/reports/REPORT.md)**

> The demo runs on Render's free tier. The first request after a period of inactivity can take a minute or two while the backend wakes up. Opening [the backend health check](https://at3-app-backend-latest.onrender.com/health) first wakes it up.

---

## Architecture

```
 ┌──────────────────────┐     GET /fare/prediction     ┌────────────────────────────┐
 │  Streamlit frontend  │ ───────────────────────────▶ │       FastAPI backend      │
 │  (trip inputs, UI)   │ ◀─────────────────────────── │  feature engineering       │
 └──────────────────────┘        [fare in USD]         │  + model routing by origin │
        Docker / Render                                └─────────────┬──────────────┘
                                                                     │
                     ┌────────────────────┬──────────────────────────┼────────────────────┐
                     ▼                    ▼                          ▼                    ▼
              ATL BOS CLT DEN      DFW DTW EWR IAD           JFK LAX LGA MIA      OAK ORD PHL SFO
                 XGBoost              CatBoost                    XGBoost             LightGBM
```

The backend turns the raw trip details into model features: calendar features, days until departure, median route distance from `models/median_traveldistance.csv`, and the cabin class of each leg. It then chooses the model trained for the origin airport. The frontend and backend each have their own Dockerfile and are deployed as separate services.

## API

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Project description |
| `GET` | `/health` | Health check |
| `GET` | `/fare/prediction` | Predicted total fare (USD) |

Example request:

```bash
curl "https://at3-app-backend-latest.onrender.com/fare/prediction?date=2026-12-15&time=09-30-00&startingAirport=JFK&destinationAirport=LAX&n_stops=0&cabin_Leg1=coach"
# [477.99]
```

| Parameter | Format / values |
|---|---|
| `date` | `YYYY-MM-DD` |
| `time` | `HH-MM-SS` |
| `startingAirport`, `destinationAirport` | ATL, BOS, CLT, DEN, DFW, DTW, EWR, IAD, JFK, LAX, LGA, MIA, OAK, ORD, PHL, SFO |
| `n_stops` | `0`–`3` |
| `cabin_Leg1` … `cabin_Leg4` | `coach`, `premium coach`, `business`, `first` (`cabin_Leg2`–`4` default to `no_stop`) |

## Running locally

Both images are built from the repository root, so they can reach `models/`:

```bash
# Backend: http://localhost:8000/docs
docker build -f app/backend/Dockerfile -t flight-fare-backend .
docker run -p 8000:8000 flight-fare-backend

# Frontend: http://localhost:8501
docker build -f app/frontend/Dockerfile -t flight-fare-frontend .
docker run -p 8501:8501 flight-fare-frontend
```

The frontend calls the deployed backend by default. To use a local backend, change `backend_url` in `app/frontend/main_frontend.py`.

## Repository structure

```
├── app/
│   ├── backend/      # FastAPI service: main.py, Dockerfile, requirements.txt
│   └── frontend/     # Streamlit UI: main_frontend.py, Dockerfile, requirements.txt
├── models/           # Trained pipelines and the route-distance lookup table
└── pyproject.toml
```

## Model performance

| Origin airports | Model | Test RMSE |
|---|---|---|
| ATL, BOS, CLT, DEN | XGBoost | $86.22 |
| DFW, DTW, EWR, IAD | CatBoost | $87.72 |
| JFK, LAX, LGA, MIA | XGBoost | $90.75 |
| OAK, ORD, PHL, SFO | LightGBM | $100.45 |

The median fare is about $345. See the [research repo](https://github.com/RatanaSovann/flight_fare_prediction) for data preparation and experiments.

## Credits

I built the app as part of a four-person group project for *36120 Advanced Machine Learning Application* at the University of Technology Sydney (2024). It integrates models trained by Ratana Sovann, Farhan Faiyaz, Yee Ting Li and Md Sadman Sakib.

Docker images: [backend](https://hub.docker.com/r/ratana001/at3_app-backend) · [frontend](https://hub.docker.com/r/ratana001/at3_app-frontend)
