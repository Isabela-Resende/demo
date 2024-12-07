import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from scipy import stats
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from scipy.special import inv_boxcox
from scipy.stats import boxcox_normplot
from streamlit.logger import get_logger

# Streamlit logger setup
LOGGER = get_logger(__name__)

# Streamlit App setup
def run():
    st.set_page_config(page_title="Income Prediction Based on Expenses")
    st.title('Income Prediction Using Regression')

    # File path input
    file_path = st.text_input("Enter the path to the data file (CSV)", "data.csv")

    if uploaded_file is not None:
       try:
        # Load data
        dataFrame = pd.read_csv(file_path)
        st.write(dataFrame)
    
         # Clean the data (remove duplicates)
        dataFrame.drop_duplicates(keep='first', inplace=True)

            # Check if required columns exist
            required_columns = ['Income', 'Age', 'Occupation', 'City_Tier', 'Rent', 'Loan_Repayment', 'Insurance', 
                                'Groceries', 'Transport', 'Eating_Out', 'Entertainment', 'Utilities', 'Healthcare', 
                                'Education', 'Dependents', 'Miscellaneous', 'Desired_Savings_Percentage', 'Desired_Savings', 
                                'Disposable_Income', 'Potential_Savings_Groceries', 'Potential_Savings_Transport', 
                                'Potential_Savings_Eating_Out', 'Potential_Savings_Entertainment', 
                                'Potential_Savings_Utilities', 'Potential_Savings_Healthcare', 
                                'Potential_Savings_Education', 'Potential_Savings_Miscellaneous']
            
            # Check if all required columns are present
            missing_columns = [col for col in required_columns if col not in dataFrame.columns]
            if missing_columns:
                st.error(f"Missing required columns: {', '.join(missing_columns)}")
                return

            # Feature Encoding using LabelEncoder
            label_encoder_occupation = LabelEncoder()
            label_encoder_city_tier = LabelEncoder()

            # Encode 'Occupation' and 'City_Tier' columns
            dataFrame['Occupation_encode'] = label_encoder_occupation.fit_transform(dataFrame['Occupation'])
            dataFrame['City_Tier_encode'] = label_encoder_city_tier.fit_transform(dataFrame['City_Tier'])

            # Transform the 'Income' variable using Box-Cox transformation
            dataFrame['Income_transform'], lambda_value = stats.boxcox(dataFrame['Income'])

            # Define X (features) and y (target)
            X = dataFrame.drop(['Occupation', 'City_Tier', 'Income', 'Income_transform'], axis=1)
            y = dataFrame['Income_transform']

            # Split the dataset into X_train, X_test, y_train, and y_test, with 10% of the data for testing
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

            # Instantiate a linear regression model
            linear_model = LinearRegression()

            # Fit the model using the training data
            linear_model.fit(X_train, y_train)

            # For each record in the test set, predict the y value (transformed value of income)
            y_pred = linear_model.predict(X_test)

            # Streamlit section for income prediction
            st.write('## Predict Your Own Income')

            # User input for features
            user_inputs = {}
            numerical_features = ['Age', 'Rent', 'Loan_Repayment', 'Insurance', 'Groceries', 'Transport', 'Eating_Out', 
                                  'Entertainment', 'Utilities', 'Healthcare', 'Education', 'Miscellaneous', 
                                  'Desired_Savings_Percentage', 'Desired_Savings', 'Disposable_Income', 'Dependents', 
                                  'Potential_Savings_Groceries', 'Potential_Savings_Transport', 'Potential_Savings_Eating_Out',
                                  'Potential_Savings_Entertainment', 'Potential_Savings_Utilities', 'Potential_Savings_Healthcare',
                                  'Potential_Savings_Education', 'Potential_Savings_Miscellaneous']
            
            for feature in numerical_features:
                user_inputs[feature] = st.slider(feature, min_value=int(dataFrame[feature].min()), max_value=int(dataFrame[feature].max()), 
                                                 value=int(dataFrame[feature].mean()))
            
            Occupation = st.selectbox('Occupation', ['Self-Employed', 'Student', 'Retired'])
            City_Tier = st.selectbox('City_Tier', ['Tier_1', 'Tier_2', 'Tier_3'])
            
            # Encode categorical variables
            Occupation_encode = label_encoder_occupation.transform([Occupation])[0]
            City_Tier_encode = label_encoder_city_tier.transform([City_Tier])[0]

            # Predict income (transformed) using the trained model
            predicted_Income_transformed = linear_model.predict([[
                user_inputs[feature] for feature in numerical_features] + [Occupation_encode, City_Tier_encode]
            ])

            # Clip the predictions to ensure they don't exceed a reasonable range
            min_value = 1e-5  # A small value to avoid negative or zero values
            max_value = 1e5   # Set a reasonable upper limit for predictions

            # Clip the transformed predictions before applying the inverse Box-Cox transformation
            predicted_Income_transformed = np.clip(predicted_Income_transformed, min_value, max_value)

            # Now apply the inverse Box-Cox transformation
            predicted_Income_transformed = inv_boxcox(predicted_Income_transformed, lambda_value)

            # Ensure the predicted income is positive (you might want to log or alert the user)
            predicted_Income_transformed = max(predicted_Income_transformed[0], 0)

            # Display prediction
            st.write(f'Predicted Income: ₹{round(predicted_Income_transformed, 0)}')

        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    run()

