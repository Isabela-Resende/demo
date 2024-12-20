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

        # Section for numerical plots
        st.write('### Display Numerical Plots')
        feature_to_plot = st.selectbox('Select a numerical feature to plot', dataFrame.select_dtypes(include=[np.number]).columns)
        if feature_to_plot:
            st.write(f'Distribution of {feature_to_plot}:')
            fig = plt.figure(figsize=(10, 6))
            plt.hist(dataFrame[feature_to_plot], bins=30, color='skyblue', edgecolor='black')
            plt.xlabel(feature_to_plot)
            plt.ylabel('Count')
            st.pyplot(fig)

        # Section for categorical plots
        st.write('### Display Categorical Plots')
        feature_to_plot = st.selectbox('Select a categorical feature to plot', ['Occupation', 'City_Tier'])
        if feature_to_plot:
            st.write(f'Distribution of {feature_to_plot}:')
            bar_chart = st.bar_chart(dataFrame[feature_to_plot].value_counts())

        # Section for relationship plots
        st.write('### Display Relationships')
        x_variable = st.selectbox('Select x-axis variable:', dataFrame.columns)
        y_variable = st.selectbox('Select y-axis variable:', dataFrame.columns)
        color_variable = st.selectbox('Select color variable:', dataFrame.columns)
        size_variable = st.selectbox('Select size variable:', dataFrame.columns)

        # Scatter plot with Plotly Express
        fig = px.scatter(dataFrame, x=x_variable, y=y_variable, color=color_variable, size=size_variable, hover_data=[color_variable])
        st.plotly_chart(fig)

        # Encode 'Occupation' and 'City_Tier' columns
        dataFrame['Occupation_encode'] = LabelEncoder().fit_transform(dataFrame['Occupation'])
        dataFrame['City_Tier_encode'] = LabelEncoder().fit_transform(dataFrame['City_Tier'])

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
        Occupation_encode = 1 if Occupation == 'Student' else (2 if Occupation == 'Self-Employed' else 0)
        City_Tier_encode = 1 if City_Tier == 'Tier_1' else (2 if City_Tier == 'Tier_2' else 0)

        # Predict income
        predicted_Income_transformed = linear_model.predict([[
            user_inputs[feature] for feature in numerical_features] + [Occupation_encode, City_Tier_encode]
        ])

        # Reverse Box-Cox transformation
        predicted_Income_transformed = inv_boxcox(predicted_Income_transformed, lambda_value)

        # Display prediction
        st.write(f'Predicted Income: {round(predicted_Income_transformed[0], 0)}')

    except Exception as e:
        st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    run()
