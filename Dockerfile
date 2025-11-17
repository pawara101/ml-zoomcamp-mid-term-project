FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

# Use production env
ENV FLASK_ENV=production

EXPOSE 5000

# Change the module path if your Flask app lives elsewhere or the app variable is named differently.
# This assumes the Flask app instance is `app` in model/predict.py
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "3", "model.predict:app"]