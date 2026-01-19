# Self_Bank.py

# Importing necessary libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline  # Uncomment if running in Jupyter

bank = pd.read_csv("Bank customers.csv")
bank.head(5)

bank.shape

bank.isnull().sum()

# Understanding about some columns first.
sns.set_theme(style="whitegrid")
sns.boxplot(bank['Customer_Age'])

# So most of the customers are somewhere near 45 age, mean being 45 with some outliers that are depicted by the dots on the right
# hand side at age 70 and maybe 75
bank[['Gender','Credit_Limit']].groupby('Gender').agg(['mean','count'])

bank[['Gender','Avg_Utilization_Ratio']].groupby('Gender').agg(['mean','count'])

# Average Utilization Ratio: it's how much you currently owe divided by your credit limit. It is generally expressed as a percent. 
# For example, if you have a total of $10,000 in credit available on two credit cards, and a balance of $5,000 on one, your credit 
# utilization rate is 50% — you're using half of the total credit you have available. A low credit utilization rate shows you're
# using less of your available credit.
bank_cards = bank.groupby("Card_Category")
bank_cards['Customer_Age'].max()

bank_cards['Customer_Age'].min()

bank_cards['Avg_Utilization_Ratio'].mean()

bank_marital = bank.groupby("Marital_Status")
bank_marital['Card_Category'].value_counts()

# Now lets move forward and see if we have categorical data in our dataset.
bank.head(3)

bank['Attrition_Flag'].value_counts()

# Note here that Attrition Flag has a value Attrited Customer, meaning this particular customer has already closed his/her
# account and is not associated with the bank as of now.
# Now for predicting the card classes, we can remove the people/customers who have attrited or keep it as it is, its your choice.

bank['Gender'].value_counts()

bank['Education_Level'].value_counts()

bank['Marital_Status'].value_counts()

def ref1(x):
    if x == 'M':
        return 1
    else:
        return 0

bank['Gender'] = bank['Gender'].map(ref1)

def ref2(x):
    if x == 'Existing Customer':
        return 1
    else:
        return 0

bank['Attrition_Flag'] = bank['Attrition_Flag'].map(ref2)

# Converting Categorical data into numerical data.
y = bank['Card_Category']
X = bank.copy()
X.head(3)

# Extracting data from the Income_Category column
X['Income_Category'].value_counts()

from sklearn.preprocessing import LabelEncoder

def label_encoded(feat):
    le = LabelEncoder()
    le.fit(feat)
    print(feat.name, le.classes_)
    return le.transform(feat)

X['Income_Category'] = label_encoded(X['Income_Category'])
X['Education_Level'] = label_encoded(X['Education_Level'])
X['Marital_Status'] = label_encoded(X['Marital_Status'])
X.head(3)

X = X.drop(['CLIENTNUM', 'Card_Category'], axis=1)

from sklearn.decomposition import PCA

pca = PCA(n_components=7)
pca2 = PCA(n_components=10)
pca_fit = pca.fit_transform(X)
pca_fit2 = pca2.fit_transform(X)

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

# You can use both of the reduced dataset to see how much accuracy or predictive power of the model is affected.
Xtrain, Xtest, ytrain, ytest = train_test_split(pca_fit2, y, test_size=0.2, random_state=42)

random_model = RandomForestClassifier(n_estimators=300, n_jobs=-1)

# Fit
random_model.fit(Xtrain, ytrain)
y_pred = random_model.predict(Xtest)

# Checking the accuracy
random_model_accuracy = round(random_model.score(Xtrain, ytrain)*100, 2)
print(round(random_model_accuracy, 2), '%')

random_model_accuracy1 = round(random_model.score(Xtest, ytest)*100, 2)
print(round(random_model_accuracy1, 2), '%')

# Save the trained model as a pickle string.
import pickle
saved_model = pickle.dump(random_model, open('BankCards.pickle', 'wb'))
saved_pca = pickle.dump(pca2, open('BankCardsPCA.pickle', 'wb'))
