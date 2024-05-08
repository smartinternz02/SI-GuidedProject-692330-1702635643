import pickle
import numpy as np
from flask import Flask, render_template, request
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import StandardScaler

app=Flask(__name__,template_folder='templates')

# Load the model
model_path = r"C:\Users\rajes\Downloads\Auto Insurance Fraud Detection\Auto Insurance Fraud Detection\py\models\dtc_model.pkl"
with open(model_path, "rb") as model_file:
    model = pickle.load(model_file)

# Load the scaler
scaler_path = r"C:\Users\rajes\Downloads\Auto Insurance Fraud Detection\Auto Insurance Fraud Detection\py\models\std_scaler.pkl"
with open(scaler_path, "rb") as scaler_file:
    scaler = pickle.load(scaler_file)
@app.route('/')
def welcome():
    return render_template('home.html')   
@app.route('/home')
def home():
    return render_template('home.html')
@app.route('/index')
def index():
    return render_template('index.html')
@app.route('/about')
def about():
    return render_template('about.html')
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        months_as_customer = float(request.form["months_as_customer"])
        policy_number = float(request.form['policy_number'])
        policy_bind_date = request.form['policy_bind_date']  # Assuming you handle date parsing separately
        policy_state = float(request.form['policy_state'])
        policy_csl = float(request.form['policy_csl'])
        policy_deductable = float(request.form['policy_deductable'])
        policy_annual_premium = float(request.form['policy_annual_premium'])
        insured_zip = float(request.form['Insured_zip'])
        insured_sex = float(request.form['insured_sex'])
        insured_occupation = float(request.form["insured_occupation"])
        insured_hobbies = float(request.form["insured_hobbies"])
        insured_relationship = float(request.form['insured_relationship'])
        capital_gains = float(request.form['capital_gains'])
        capital_loss = float(request.form['capital_loss'])
        incident_date = float(request.form['incident_date'])  # Assuming you handle date parsing separately
        incident_type = float(request.form['incident_type'])
        collision_type = float(request.form['collision_type'])
        incident_severity = float(request.form['incident_severity'])
        authorities_contacted = float(request.form['authorities_contacted'])
        incident_location = float(request.form['incident_location'])
        incident_hour_of_the_day = float(request.form['incident_hour_of_the_day'])
        number_of_vehicles_involved = float(request.form['number_of_vehicles_involved'])
        property_damage = float(request.form['property_damage'])
        bodily_injuries = float(request.form['bodily_injuries'])
        witnesses = float(request.form['witnesses'])
        police_report_available = float(request.form['police_report_available'])
        total_claim_amount = float(request.form['total_claim_amount'])
        auto_make = float(request.form['auto_make'])
        auto_model = float(request.form['auto_model'])
        auto_year = float(request.form['auto_year'])
        form_data = request.form
        input_data = []
        for field in form_data:
            input_data.append(float(form_data[field]))

        # Check if the scaler has been fitted
        if not hasattr(scaler, 'mean_'):
            raise NotFittedError("Scaler is not fitted. Please fit the scaler on training data before using it for prediction.")

        # Scale the input features
        input_data = np.array([[months_as_customer, policy_number, policy_state, policy_csl, policy_deductable,policy_bind_date, policy_annual_premium, insured_zip, insured_sex, insured_occupation, insured_hobbies,capital_gains, capital_loss,incident_date, incident_type, collision_type, incident_severity, authorities_contacted, incident_location, incident_hour_of_the_day, number_of_vehicles_involved, property_damage, bodily_injuries, witnesses, police_report_available, total_claim_amount, auto_make, auto_year]])
        scaled_input = scaler.transform(input_data)

        # Make prediction
        prediction = model.predict(scaled_input)
        prediction = "Legal Insurance Claim" if prediction == 0 else "Fraud Insurance Claim"

        # Render prediction template
        return render_template('prediction.html', prediction=prediction)

    except NotFittedError as e:
        # Handle case where scaler is not fitted
        print(e)
        return render_template('prediction.html', prediction="Error: Scaler not fitted. Please fit the scaler on training data before using it for prediction.")

if __name__ == '__main__':
    app.run(debug=True)
