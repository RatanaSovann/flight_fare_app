FRONTEND RENDER LINK:
https://at3-app-frontend-latest.onrender.com/

BACKEND RENDER LINK:
https://at3-app-backend-latest.onrender.com/


To smoothly run the app it is recommended to buffer the backend first before running the frontend (this might take a few minutes)


LINK TO EXPERIMENT GITHUB REPO:
https://github.com/FarhanFaiyaz/adv_mla_at3


The App is split to backend and frontend, with different Dockerfiles and requirement.txt. 

The folder has the following structure:

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


Note: The frontend and backend folders and Docker container structure is set from the root at3_app. This is done so that both the backend and frontend have access to the models folder. Requirement.txt for frontend also incorporate the functionality to load the model directly into main.py. However, to keep the code clean we import request to get prediction from FastAPI(backend) where feature engineering of inputs and transformation takes place.  

To see the latest docker image pushed:

LINK TO BACKEND DOCKERHUB: 
https://hub.docker.com/repository/docker/ratana001/at3_app-backend/general

LINK TO FRONTEND DOCKERHUB:
https://hub.docker.com/repository/docker/ratana001/at3_app-frontend/general

