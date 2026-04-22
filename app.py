from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load model & scaler
model = joblib.load("loan_model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from form
    age = float(request.form['Age'])
    income = float(request.form['Income'])
    loan_amount = float(request.form['Loan_Amount'])
    credit_score = float(request.form['Credit_Score'])
    emp_years = float(request.form['Employment_Years'])
    edu = int(request.form['Education_Level'])
    house = int(request.form['Housing_Status'])

    data = np.array([[age, income, loan_amount, credit_score, emp_years, edu, house]])

    # Scaling
    data = scaler.transform(data)

    prediction = model.predict(data)

    if prediction[0] == 0:
        result = "Eligible for Loan ✅"
    else:
        result = "Not Eligible ❌"

    return render_template('index.html', prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True)

if __name__ == "__main__":
    app.run(debug=True)