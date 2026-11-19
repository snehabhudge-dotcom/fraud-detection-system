import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load cleaned dataset
df = pd.read_csv("dataset/cleaned_creditcard.csv")

# Graph 1: Normal vs Fraud Transactions
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="Class")
plt.title("Normal vs Fraud Transactions")
plt.xlabel("Class (0 = Normal, 1 = Fraud)")
plt.ylabel("Number of Transactions")
plt.show()

# Graph 2: Transaction Amount Distribution
plt.figure(figsize=(7, 4))
sns.histplot(df["Amount"], bins=50)
plt.title("Transaction Amount Distribution")
plt.xlabel("Transaction Amount")
plt.ylabel("Frequency")
plt.show()

# Graph 3: Amount by Transaction Class
plt.figure(figsize=(6, 4))
sns.boxplot(data=df, x="Class", y="Amount")
plt.title("Transaction Amount by Class")
plt.show()

# Graph 4: Correlation Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), cmap="coolwarm", center=0)
plt.title("Correlation Heatmap")
plt.show()

print("EDA Completed Successfully!")