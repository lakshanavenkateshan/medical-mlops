# Heart Disease Risk Prediction (MLOps)
End-to-end ML pipeline using FastAPI + Docker  
## Features
- Data preprocessing  
- Model training & saving  
- FastAPI prediction API  
- Docker containerization  
## Run locally
docker build -t medical-mlops-app .
docker run -p 8000:8000 medical-mlops-app
