import requests

url = 'http://127.0.0.1:5000/predict'

customer = {
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
}

prediction = requests.post(url, json=customer).json()