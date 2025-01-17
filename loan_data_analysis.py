import pandas as pd


loan_data = pd.read_csv('loans_data.csv')
loan_data.info()
print(loan_data.isnull().any())