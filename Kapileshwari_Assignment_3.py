# ASSIGMENT 3
# Name-Anirudha Kapileshwari
# Spring 2024 Introduction to Machine Learning (CS-484-01)
# prof Shouvik Roy


#Problem 1: Linear Regression (13 marks)
#The dataset lab03_dataset_1.csv has 6,435 rows of data pertaining to Walmart sales and employment. The input features are Store, Weekly_Sales, Holiday_Flag, Temperature, Fuel_Price, CPI and the output is Unemployment. Perform the following tasks:
#1. Create a heatmap for the entire dataset. (1 mark)
#2. The input features should be subjected to feature scaling, specifically the min-max
#scaling. (2 marks)
#3. Once the scaled input features are ready, learn a model using sklearn’s linear
#regression module. Use a 90-10 train-test split for the learning process. (2 marks)
#4. After you generate the linear regression model, output the regression score,
#coefficients, intercept and mean squared error (over the test set). (5 marks)
#5. Create a scatter plot which showcases the true output and the predicted output for the test case. Make sure to display a single plot which should contain both the data points. Use two different colors to represent the two types of data. Don’t forget to
#add a legend to the plot. (3 marks)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the dataset
dataset = pd.read_csv('lab03_dataset_1.csv')

# Task 1: Create a heatmap for the entire dataset
def plot_heatmap(data):
    plt.figure(figsize=(10, 8))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Heatmap of the Dataset')
    plt.show()

# Task 2: Perform min-max scaling on input features
def min_max_scaling(data):
    # Extract input features
    features = data.drop(columns=['Unemployment'])
    
    # Initialize MinMaxScaler
    scaler = MinMaxScaler()
    
    # Fit and transform the input features
    scaled_features = scaler.fit_transform(features)
    
    # Convert scaled_features back to DataFrame
    scaled_df = pd.DataFrame(scaled_features, columns=features.columns)
    
    return scaled_df, scaler

# Task 3: Train Linear Regression model
def train_linear_regression(X_train, y_train):
    # Initialize Linear Regression model
    model = LinearRegression()
    
    # Train the model
    model.fit(X_train, y_train)
    
    return model

# Task 4: Predict on test set and evaluate
def predict_and_evaluate(model, X_test, y_test):
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
    
    # Plot true vs. predicted values
    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(y_pred)), y_pred, color='blue', label='Predicted')
    plt.scatter(range(len(y_test)), y_test, color='red', label='True')
    plt.title('True vs. Predicted Unemployment Rates')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.legend()
    plt.grid(True)
    plt.show()

# Main function
def main():
    plot_heatmap(dataset)
    
    scaled_df, scaler = min_max_scaling(dataset)
    
    # Splitting the data into train and test sets (90-10 split)
    X_train, X_test, y_train, y_test = train_test_split(scaled_df, dataset['Unemployment'], test_size=0.1, random_state=42)
    
    model = train_linear_regression(X_train, y_train)
    
    predict_and_evaluate(model, X_test, y_test)

if __name__ == "__main__":
    main()


####################################################################################################################################
####################################################################################################################################
#Problem 2: k – Nearest Neighbors (12 marks)
#The dataset lab03_dataset_2.csv has the results of fraud investigations of 5,960 cases. The binary variable FRAUD indicates the result (output class) with 1 = Fraud, 0 = Not Fraud. The other quantitative variables contain information about the cases.
#• DOCTOR_VISITS: Number of visits to a doctor.
#• MEMBER_DURATION: Membership duration in number of months.
#• NUM_CLAIMS: Number of claims made recently.
#• NUM_MEMBERS: Number of members covered.
#• OPTOM_PRESC: Number of optical examinations.
#• TOTAL_SPEND: Total amount of claims in dollars.
#Use the first 20% of the dataset i.e., the first 20% of the rows as the test set, while the remaining bottom 80% rows will be your training set. During majority voting, if both the classes have equal distribution within the nearest neighborhood, choose class = 1 (Fraud).
#1. The input features used during training should be subjected to feature scaling, specifically the min-max scaling. (2 marks)
#2. You will use sklearn’s k – nearest neighbors module to learn a classification model with multiple nearest neighbors ranging from 2 to 5. Apply the learned k–NN model to classify the test set. Compute the misclassification rates for k ranging from 2 to 5. Use Euclidean distance as the similarity measure. (5 marks)
#3. Next, apply sklearn’s k–d tree module to classify the test set. In a similar manner to the above scenario, compute the misclassification rates for k ranging from 2 to 5. Use Manhattan distance as the similarity measure. (5 marks)


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Step 1: Load the dataset and split it into training and testing sets
def load_and_split_data():
    data = pd.read_csv('lab03_dataset_2.csv')
    X = data.drop(columns=['FRAUD'])
    y = data['FRAUD']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# Step 2: Apply feature scaling (min-max scaling)
def apply_feature_scaling(X_train, X_test):
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

# Steps 3 and 4: Train k-NN models and compute misclassification rates for k values ranging from 2 to 5 (Euclidean distance)
def train_and_evaluate_knn(X_train_scaled, X_test_scaled, y_train, y_test, distance='euclidean'):
    print(f"Using k-NN ({distance} distance):")
    for k in range(2, 6):
        knn = KNeighborsClassifier(n_neighbors=k, metric=distance)
        knn.fit(X_train_scaled, y_train)
        y_pred = knn.predict(X_test_scaled)
        misclassification_rate = 1 - accuracy_score(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"k = {k}, Misclassification rate: {misclassification_rate:.4f}, Accuracy: {accuracy:.4f}")

# Steps 5 and 6: Train k-d tree models and compute misclassification rates for k values ranging from 2 to 5 (Manhattan distance)
def train_and_evaluate_kd_tree(X_train_scaled, X_test_scaled, y_train, y_test, distance='manhattan'):
    print(f"\nUsing k-d tree ({distance} distance):")
    for k in range(2, 6):
        knn_tree = KNeighborsClassifier(n_neighbors=k, algorithm='kd_tree', metric=distance)
        knn_tree.fit(X_train_scaled, y_train)
        y_pred_tree = knn_tree.predict(X_test_scaled)
        misclassification_rate_tree = 1 - accuracy_score(y_test, y_pred_tree)
        accuracy_tree = accuracy_score(y_test, y_pred_tree)
        print(f"k = {k}, Misclassification rate: {misclassification_rate_tree:.4f}, Accuracy: {accuracy_tree:.4f}")

# Main function
def main():
    X_train, X_test, y_train, y_test = load_and_split_data()
    X_train_scaled, X_test_scaled = apply_feature_scaling(X_train, X_test)
    
    train_and_evaluate_knn(X_train_scaled, X_test_scaled, y_train, y_test, distance='euclidean')
    train_and_evaluate_kd_tree(X_train_scaled, X_test_scaled, y_train, y_test, distance='manhattan')

if __name__ == "__main__":
    main()

