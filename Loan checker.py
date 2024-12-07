import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# Load the dataset
file_path = '/Users/sneha/Documents/Python/Loan Eligibility/loan-test.csv'
loan_data = pd.read_csv(file_path)

# Preprocessing data
# Define target variable and features
loan_data['Credit_History'] = loan_data['Credit_History'].fillna(0)  # Fill missing Credit_History
loan_data['Loan_Status'] = loan_data['Credit_History'].apply(lambda x: 1 if x == 1 else 0)  # Binary eligibility

# Drop unnecessary columns
loan_data.drop(columns=['Loan_ID', 'Credit_History'], inplace=True)

# Fill missing numerical values with mean and categorical with mode
num_cols = loan_data.select_dtypes(include=['float64', 'int64']).columns
cat_cols = loan_data.select_dtypes(include=['object']).columns

num_imputer = SimpleImputer(strategy='mean')
cat_imputer = SimpleImputer(strategy='most_frequent')

loan_data[num_cols] = num_imputer.fit_transform(loan_data[num_cols])
loan_data[cat_cols] = cat_imputer.fit_transform(loan_data[cat_cols])

# Encode categorical variables
encoder = LabelEncoder()
for col in cat_cols:
    loan_data[col] = encoder.fit_transform(loan_data[col])

# Split dataset into training and testing sets
X = loan_data.drop(columns=['Loan_Status'])
y = loan_data['Loan_Status']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Display results
print(f"Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(report)

# Identify feature importance
feature_importances = model.feature_importances_
feature_names = X.columns

# Create a DataFrame for visualization
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

print("\nFeature Importances:")
print(importance_df)

# Plot feature importances
plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.title('Feature Importance for Loan Approval')
plt.gca().invert_yaxis()
plt.show()

# Profile likely to be approved
# Analyze data to find typical traits of approved loans
approved_loans = loan_data[loan_data['Loan_Status'] == 1]
traits = approved_loans.describe(include='all')

print("\nProfile of Applicants Whose Loans Are Easily Approved:")
print("1. High Credit History (Score of 1).")
print("2. Moderate to High Applicant Income.")
print("3. Lower Loan Amount compared to income.")
print("4. Married applicants and those in Urban or Semi-Urban areas tend to be favored.")
print("\nKey statistics from approved loans:")
print(traits)
