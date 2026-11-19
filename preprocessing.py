import pandas as pd

# Load dataset
df = pd.read_csv("dataset/creditcard.csv")

print("Original Dataset Shape:", df.shape)

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum().sum())

# Check duplicate records
print("\nDuplicate Records:")
print(df.duplicated().sum())

# Remove duplicate records
df = df.drop_duplicates()

# Check transaction distribution
print("\nTransaction Distribution:")
print(df["Class"].value_counts())

# Save cleaned dataset
df.to_csv("dataset/cleaned_creditcard.csv", index=False)

print("\nPreprocessing Completed Successfully!")
print("Cleaned Dataset Shape:", df.shape)