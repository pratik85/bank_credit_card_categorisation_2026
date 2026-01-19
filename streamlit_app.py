import streamlit as st
import numpy as np
import pickle

# Load trained model and PCA
with open('BankCards.pickle', 'rb') as f:
    model = pickle.load(f)
with open('BankCardsPCA.pickle', 'rb') as f:
    pca = pickle.load(f)

st.title('Bank Card Category Prediction')
st.write('Enter customer details to predict the card category.')

# User input form
def user_input_features():
    Customer_Age = st.number_input('Customer Age', min_value=18, max_value=100, value=45)
    Gender = st.selectbox('Gender', ['M', 'F'])
    Dependent_count = st.number_input('Dependent Count', min_value=0, max_value=10, value=1)
    Education_Level = st.selectbox('Education Level', ['Unknown', 'Uneducated', 'High School', 'College', 'Graduate', 'Post-Graduate', 'Doctorate'])
    Marital_Status = st.selectbox('Marital Status', ['Unknown', 'Single', 'Married', 'Divorced'])
    Income_Category = st.selectbox('Income Category', ['Unknown', 'Less than $40K', '$40K - $60K', '$60K - $80K', '$80K - $120K', '$120K +'])
    Card_Category = st.selectbox('Card Category (for reference)', ['Blue', 'Silver', 'Gold', 'Platinum'])
    Months_on_book = st.number_input('Months on Book', min_value=0, max_value=100, value=39)
    Total_Relationship_Count = st.number_input('Total Relationship Count', min_value=0, max_value=10, value=3)
    Months_Inactive_12_mon = st.number_input('Months Inactive (12 mon)', min_value=0, max_value=12, value=2)
    Contacts_Count_12_mon = st.number_input('Contacts Count (12 mon)', min_value=0, max_value=10, value=3)
    Credit_Limit = st.number_input('Credit Limit', min_value=0.0, value=5000.0)
    Total_Revolving_Bal = st.number_input('Total Revolving Bal', min_value=0, value=1000)
    Avg_Open_To_Buy = st.number_input('Avg Open To Buy', min_value=0.0, value=4000.0)
    Total_Amt_Chng_Q4_Q1 = st.number_input('Total Amt Chng Q4 Q1', min_value=0.0, value=1.0)
    Total_Trans_Amt = st.number_input('Total Trans Amt', min_value=0, value=5000)
    Total_Trans_Ct = st.number_input('Total Trans Ct', min_value=0, value=40)
    Total_Ct_Chng_Q4_Q1 = st.number_input('Total Ct Chng Q4 Q1', min_value=0.0, value=1.0)
    Avg_Utilization_Ratio = st.number_input('Avg Utilization Ratio', min_value=0.0, max_value=1.0, value=0.2)
    Attrition_Flag = st.selectbox('Attrition Flag', ['Existing Customer', 'Attrited Customer'])
    
    # Map categorical to numerical as in training
    Gender = 1 if Gender == 'M' else 0
    Attrition_Flag = 1 if Attrition_Flag == 'Existing Customer' else 0
    # Label encoding for Education_Level, Marital_Status, Income_Category
    education_map = {'Unknown':0, 'Uneducated':1, 'High School':2, 'College':3, 'Graduate':4, 'Post-Graduate':5, 'Doctorate':6}
    marital_map = {'Unknown':0, 'Single':1, 'Married':2, 'Divorced':3}
    income_map = {'Unknown':0, 'Less than $40K':1, '$40K - $60K':2, '$60K - $80K':3, '$80K - $120K':4, '$120K +':5}
    Education_Level = education_map[Education_Level]
    Marital_Status = marital_map[Marital_Status]
    Income_Category = income_map[Income_Category]
    
    data = np.array([
        Customer_Age, Gender, Dependent_count, Education_Level, Marital_Status, Income_Category,
        Months_on_book, Total_Relationship_Count, Months_Inactive_12_mon, Contacts_Count_12_mon,
        Credit_Limit, Total_Revolving_Bal, Avg_Open_To_Buy, Total_Amt_Chng_Q4_Q1, Total_Trans_Amt,
        Total_Trans_Ct, Total_Ct_Chng_Q4_Q1, Avg_Utilization_Ratio, Attrition_Flag
    ]).reshape(1, -1)
    return data

input_data = user_input_features()

# Drop CLIENTNUM and Card_Category as in training
# Apply PCA
input_data_pca = pca.transform(input_data)

if st.button('Predict Card Category'):
    prediction = model.predict(input_data_pca)
    st.success(f'Predicted Card Category: {prediction[0]}')
