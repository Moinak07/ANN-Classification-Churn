
# Customer Churn Prediction

A brief description of what this project does and who it's for

Customer Churn Prediction
This is a Customer Churn Prediction application built using Streamlit and a pre-trained TensorFlow model. It allows users to input customer details and predicts the likelihood of a customer leaving the bank.

Features
 Intuitive user interface for entering customer details.
 Predicts the probability of customer churn using a trained model.
 Shows detailed input data alongside prediction output.
 Pre-trained model, encoders, and scaler for real-time prediction.




Installation
1.Clone the repository
git clone https://github.com/your-repo/customer-churn-prediction.git
cd customer-churn-prediction

2.Install dependencies
pip install -r requirements.txt


3.Ensure the following files are present in the directory:

model.h5
label_encoder_gender.pkl
onehot_encoder_geo.pkl
scaler.pkl

4.Usage
Run the application using Streamlit:
streamlit run app.py

Input Fields
Geography: Customer's country (e.g., France, Germany, Spain).

Gender: Male / Female.

Age: Customer's age.

Account Balance: Bank account balance.

Credit Score: Credit score (0 - 850).

Estimated Salary: Customer's salary.

Tenure: Years with the bank.

Number of Products: Bank products used (1-4).

Has Credit Card: Yes / No (0 / 1).

Active Member: Yes / No (0 / 1).


Prediction Output
Churn Probability: A float value between 0 and 1.

Result: A message indicating if the customer is likely to chur

Dependencies
This project uses the following Python libraries:

streamlit

tensorflow

numpy

pandas

scikit-learn

pickle (for loading pre-trained encoders and scaler)

🧠 Model Details
The model (model.h5) is a neural network trained using the Churn_Modelling.csv dataset. It uses various features (e.g., credit score, geography, age) to predict the likelihood of customer churn.

🛠 How It Works
Input Data Preparation:

Collect input via Streamlit UI.

Encode categorical variables using pre-trained encoders.

Scale the features using a pre-trained scaler.

Prediction:

Input the processed data to the trained model.

Get the churn probability.

Output:

Show the result in a user-friendly format on the web app.

📚 Logs
You can store and visualize training logs in a logs/ directory using tools like TensorBoard (optional).

📄 License
This project is licensed under the MIT License.
See the LICENSE file for more details.

🙏 Acknowledgments
Dataset: [Churn_Modelling.csv]

Frameworks: Streamlit, TensorFlow, Scikit-learn

