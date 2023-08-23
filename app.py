import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

# Load the pre-trained model
try:
    model = pickle.load(open('churn.pkl', 'rb'))
except FileNotFoundError:
    print("Error: 'churn.pkl' file not found.")
except Exception as e:
    print("An error occurred while loading the model:", str(e))

# List of features for input and its length
X = ['Tenure', 'WarehouseToHome', 'NumberOfDeviceRegistered',
       'NumberOfAddress', 'DaySinceLastOrder', 'CashbackAmount',
       'PreferredLoginDevice_Computer', 'PreferredLoginDevice_Mobile',
       'CityTier_1', 'CityTier_2', 'CityTier_3', 'PreferredPaymentMode_CC',
       'PreferredPaymentMode_COD', 'PreferredPaymentMode_DC',
       'PreferredPaymentMode_E wallet', 'PreferredPaymentMode_UPI',
       'PreferedOrderCat_Fashion', 'PreferedOrderCat_Grocery',
       'PreferedOrderCat_Laptop', 'PreferedOrderCat_Mobile',
       'PreferedOrderCat_Others', 'SatisfactionScore_1', 'SatisfactionScore_2',
       'SatisfactionScore_3', 'SatisfactionScore_4', 'SatisfactionScore_5',
       'MaritalStatus_Divorced', 'MaritalStatus_Married',
       'MaritalStatus_Single', 'Gender_Female', 'Gender_Male', 'Complain_0',
       'Complain_1']
len(X)

def predict_price(tenure, warehouse, numdevice, numaddress, lastorder, cashback, logindevice, citytier, paymentmode, ordercat, score, maritalstatus, gender, complain):    
    # Extract indices for categorical variables from the feature list
    logindevice_index = X.index('PreferredLoginDevice_' + logindevice)
    citytier_index = X.index('CityTier_' + citytier)
    paymentmode_index = X.index('PreferredPaymentMode_' + paymentmode)
    ordercat_index = X.index('PreferedOrderCat_' + ordercat)
    score_index = X.index('SatisfactionScore_' + score)
    maritalstatus_index = X.index('MaritalStatus_' + maritalstatus)
    gender_index = X.index('Gender_' + gender)
    complain_index = X.index('Complain_' + complain)

    index_list = [logindevice_index, citytier_index, paymentmode_index, ordercat_index, score_index, maritalstatus_index, gender_index, complain_index]

    x = np.zeros(len(X))
    # Assign values to input features
    x[0] = tenure
    x[1] = warehouse
    x[2] = numdevice
    x[3] = numaddress
    x[4] = lastorder
    x[5] = cashback

    # Set values to 1 for categorical features based on extracted indices
    for ind in index_list:
        if ind >= 0:
            x[ind] = 1

    return model.predict([x])[0]

# Create a Flask web application instance
app = Flask(__name__)

# Define the route for the homepage
@app.route('/')
def home():
    return render_template('index.html')

# Define the route for the prediction and form submission
@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    if request.method == 'POST':
        # Extract form input values
        tenure = float(request.form['tenure'])
        warehouse = float(request.form['warehousetohome'])
        numdevice = float(request.form['numdevices'])
        numaddress = float(request.form['numaddress'])
        lastorder = float(request.form['lastorder'])
        cashback = float(request.form['cashbackamount'])
        logindevice = request.form['logindevice']
        citytier = request.form['citytier']
        paymentmode = request.form['paymentmode']
        ordercat = request.form['ordercat']
        score = request.form['satisfactionscore']
        maritalstatus = request.form['maritalstatus']
        gender = request.form['gender']
        complain = request.form['complain']
  
    # Call the prediction function
    prediction = predict_price(tenure, warehouse, numdevice, numaddress, lastorder, cashback, logindevice, citytier, paymentmode, ordercat, score, maritalstatus, gender, complain)    
  
    return render_template('index.html', prediction=prediction)

# Run the Flask application
if __name__ == "__main__":
    app.run(debug=True)
