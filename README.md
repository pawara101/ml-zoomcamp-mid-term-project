# Credit Card Risk Approval Model

Brief project for predicting credit-card approval (binary classification). The model estimates the probability a credit application will be approved based on applicant features (age, income, credit score, employment, etc.).

## Problem description
Given applicant data, predict the likelihood of credit approval. Useful for automating initial screening and risk scoring.

## Repo structure (relevant)
- model/  
  - train.py — training script (produces serialized model and preprocessors)  
  - predict.py — Flask app that serves predictions  
  - get-predictions_test.py — example client / test caller  
- data/clean_dataset.csv — cleaned dataset used for training (expected)  
- ml-zoomcamp-1.ipynb — exploratory notebook and sample usage  
- requirements.txt — Python dependencies  
- Dockerfile — container image for deployment

## Requirements
- Linux (instructions below use bash)
- Python 3.11
- Docker (optional, for containerized deployment)

Install Python deps:
```bash
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Train the model (local)
1. Ensure `data/clean_dataset.csv` is available.
2. Train and save model artifacts:
```bash
python model/train.py
```
This should create a serialized model bundle (e.g. `model_1.bin`) in the repo (check `train.py` for exact output path).

## Run the API (local)
Start the Flask/Gunicorn server (expects the app instance in `model/predict.py`):
```bash
# development (direct flask)
python model/predict.py

# production (gunicorn)
gunicorn --bind 0.0.0.0:5000 model.predict:app
```

Note: open `model/predict.py` to confirm the route path (e.g. `/predict` or `/predict_approval`) and expected JSON schema.

## Example request
Use the sample JSON below (fields should match the model's expected input columns). Adjust names/types to match `model/predict.py`.

```bash
curl -sS -X POST "http://localhost:5000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "female",
    "age": 35.6,
    "debt": 2.15,
    "married": "yes",
    "bankcustomer": "yes",
    "industry": "Industrials",
    "ethnicity": "White",
    "yearsemployed": 4.2,
    "priordefault": "no",
    "employed": "yes",
    "creditscore": 5,
    "driverslicense": "yes",
    "citizen": "ByBirth",
    "zipcode": 180,
    "income": 2450
  }'
```

If the endpoint name differs, replace `/predict` with the route defined in `model/predict.py`.

## Docker — build and run
1. From the project root build the image:
```bash
docker build -t ml-zoomcamp-app .
```

2. If your trained model artifact is `model/model_1.bin` in the repo, you can mount it into the container at runtime (recommended so you can swap models without rebuilding):
```bash
# from repo root
docker run --rm -p 5000:5000 \
  -v "$(pwd)/model/model_1.bin:/app/model/model_1.bin:ro" \
  --env FLASK_ENV=production \
  ml-zoomcamp-app
```

3. Alternative: copy the artifact into the image before running (rebuild required if you change the model):
```bash
# ensure model/model_1.bin exists in repo, then:
docker build -t ml-zoomcamp-app .
docker run --rm -p 5000:5000 ml-zoomcamp-app
```

Notes:
- The Dockerfile runs Gunicorn with the WSGI target `model.predict:app`. If your Flask app variable or module path differs, update the Dockerfile CMD or use a custom entrypoint.
- Exposed port is 5000; map it with `-p HOST_PORT:5000`.
- Use `-v` to mount logs or data directories if needed.

## Docker troubleshooting
- Confirm `model/predict.py` loads the same path/name for the model artifact as the mounted path.
- If JSON responses fail due to NumPy types, convert arrays/scalars to native Python types before returning (use `.tolist()` or `.item()`).

## Notes and troubleshooting
- Inspect `model/predict.py` for expected input schema and response format.
- Ensure the trained model and preprocessors (e.g. OHE, scaler) are available and loaded by `predict.py`.
