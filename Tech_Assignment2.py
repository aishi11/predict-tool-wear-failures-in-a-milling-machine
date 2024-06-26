import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE

# Load the dataset
file_path = 'ai4i2020.csv'
data = pd.read_csv(file_path)

# Check for missing values
missing_values = data.isnull().sum()

# Encode categorical variables
label_encoder = LabelEncoder()
data['Type'] = label_encoder.fit_transform(data['Type'])

# Scale numerical features
scaler = StandardScaler()
numerical_features = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
data[numerical_features] = scaler.fit_transform(data[numerical_features])

# Define the target variable and features
X = data.drop(columns=['UDI', 'Product ID', 'Machine failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'])
y = data['TWF']

# Apply SMOTE to balance the classes
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Split the resampled data into training and testing sets
X_train_resampled, X_test_resampled, y_train_resampled, y_test_resampled = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Train a Logistic Regression model on the resampled data
model_resampled = LogisticRegression(random_state=42)
model_resampled.fit(X_train_resampled, y_train_resampled)

# Make predictions on the test set
y_pred_resampled = model_resampled.predict(X_test_resampled)

# Evaluate the resampled model
accuracy_resampled = accuracy_score(y_test_resampled, y_pred_resampled)
conf_matrix_resampled = confusion_matrix(y_test_resampled, y_pred_resampled)
class_report_resampled = classification_report(y_test_resampled, y_pred_resampled)

# Print the evaluation results
print(f'Accuracy: {accuracy_resampled}')
print(f'Confusion Matrix:\n{conf_matrix_resampled}')
print(f'Classification Report:\n{class_report_resampled}')
