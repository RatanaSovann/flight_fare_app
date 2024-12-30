# AT3 App: Flight Fare Prediction

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
