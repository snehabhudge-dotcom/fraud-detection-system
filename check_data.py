import pandas as pd

# Load dataset
df = pd.read_csv("dataset/creditcard.csv")

# Display first 5 rows
print(df.head())

# Display dataset information
print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nMissing Values:")
print(df.isnull().sum().sum())

print("\nFraud and Normal Transactions:")
print(df["Class"].value_counts())