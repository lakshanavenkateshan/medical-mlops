# Heart Disease Risk Prediction – MLOps Project

## Project Overview
This project demonstrates an end-to-end **Machine Learning workflow** for predicting the risk of heart disease using patient vitals (heart rate, SpO2, temperature). It includes **data generation, cleaning, model training, evaluation, artifact storage, and version control**.


---

## Project Workflow

1. **Data Generation (Synthetic Dataset)**
   - Generates sample patient data (`synthetic_data.csv`) with vitals for experimentation.
   
2. **Data Cleaning & Visualization**
   - Cleans the raw dataset.
   - Visualizes feature distributions and correlations.
   - Saves cleaned dataset as `cleaned_data.csv`.

3. **Model Training & Evaluation**
   - Trains a **Logistic Regression** model using cleaned features.
   - Normalizes features with `StandardScaler`.
   - Evaluates the model with:
     - Accuracy
     - Confusion matrix
     - Classification report
   - Saves trained model (`heart_disease_model.pkl`) and scaler (`scaler.pkl`) in `models/`.

4. **AWS Backup**
   - Uploads model and scaler artifacts to **AWS S3** for secure storage.

5. **Version Control**
   - Project tracked with **Git**, allowing collaborative development and safe versioning.

---

## Usage

### 1. Generate Synthetic Data
```bash
python src/generate_data.py
```
## clean & visualize data
```bash
python src/data_clean_visualize.py
```
## Train model
```bash
python src/train_model.py
```
### Upload Model to AWS
```bash
python src/upload_to_s3.py
```
### Git Version Control
```bash
git init
git add .
git commit -m "Initial commit / Day 36: AWS backup + model training"
git remote add origin <your-github-url>
git push -u origin main
```
### future work 
Future Work
Improve model performance with more features (e.g., age, cholesterol, blood pressure).
Try more advanced classifiers (Random Forest, XGBoost).
Deploy the model with a web dashboard using Streamlit or Flask.
Automate CI/CD pipeline for MLOps integration.
