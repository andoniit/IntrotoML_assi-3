import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the dataset
data = pd.read_csv('lab03_dataset_1.csv')


# Task 1: Create a heatmap for the entire dataset
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap of the Dataset')
plt.show()

# Task 2: Perform min-max scaling on input features
# Extract input features
input_features = data.drop(columns=['Unemployment'])

# Initialize MinMaxScaler
scaler = MinMaxScaler()

# Fit and transform the input features
scaled_features = scaler.fit_transform(input_features)

# Convert scaled_features back to DataFrame
scaled_df = pd.DataFrame(scaled_features, columns=input_features.columns)



# Splitting the data into train and test sets (90-10 split)
X_train, X_test, y_train, y_test = train_test_split(scaled_df, data['Unemployment'], test_size=0.1, random_state=42)

# Initialize Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)

# Output the regression score
regression_score = model.score(X_test, y_test)
print("Regression Score:", regression_score)

# Output the coefficients
coefficients = model.coef_
intercept = model.intercept_
print("Coefficients:", coefficients)
print("Intercept:", intercept)

# Output the mean squared error (MSE)
print("Mean Squared Error (Test Set):", mse)

plt.figure(figsize=(10, 6))
plt.scatter(range(len(y_pred)), y_pred, color='blue', label='Predicted')
plt.scatter(range(len(y_test)), y_test, color='red', label='True')
plt.title('True vs. Predicted Unemployment Rates')
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.legend()
plt.grid(True)
plt.show()