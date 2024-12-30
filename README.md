# Flight_Fare_app: Flight Fare Prediction

## Live Links

- **Frontend**: [at3-app-frontend-latest](https://at3-app-frontend-latest.onrender.com/)
- **Backend**: [at3-app-backend-latest](https://at3-app-backend-latest.onrender.com/)

> **Note**: For smooth operation, buffer the backend by opening its link first before using the frontend. This process may take a few minutes.

---

## Project Overview
This project is a flight fare prediction application split into:
1. **Backend** (FastAPI): Handles prediction requests, feature engineering, and transformations.
2. **Frontend** (Streamlit): Provides an interactive interface for users to input flight details and view predictions.

---

## Folder Structure
```plaintext
at3_app/
│
├── app/
│   ├── backend/
│   │   ├── main.py
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   │
│   ├── frontend/
│   │   ├── main_frontend.py
│   │   ├── Dockerfile
│   │   └── requirements.txt
│
└── models/
    └── xgb_pipeline.joblib

### Key Details
- **Model Access**: Both the backend and frontend can access the `models` folder for predictions.  
- **Frontend Model Loading**: Although the frontend can load models directly, predictions are fetched via API requests to the backend to maintain clean and modular code.  

---

### Repository and Docker Links

#### GitHub Repository
- **[Experiment Repository](https://github.com/FarhanFaiyaz/adv_mla_at3)**  

#### DockerHub Images
- **Backend**: [ratana001/at3_app-backend](https://hub.docker.com/repository/docker/ratana001/at3_app-backend/general)  
- **Frontend**: [ratana001/at3_app-frontend](https://hub.docker.com/repository/docker/ratana001/at3_app-frontend/general)  

---

### Additional Notes
- Ensure the backend is running before using the frontend.  
- The project uses separate `Dockerfile` and `requirements.txt` files for backend and frontend components.  

