import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Step 1: Load the dataset and split it into training and testing sets
data = pd.read_csv('lab03_dataset_2.csv')
X = data.drop(columns=['FRAUD'])
y = data['FRAUD']

# Splitting the dataset into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2: Apply feature scaling (min-max scaling)
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Steps 3 and 4: Train k-NN models and compute misclassification rates for k values ranging from 2 to 5 (Euclidean distance)
print("Using k-NN (Euclidean distance):")
for k in range(2, 6):
    knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
    knn.fit(X_train_scaled, y_train)
    y_pred = knn.predict(X_test_scaled)
    misclassification_rate = 1 - accuracy_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"k = {k}, Misclassification rate: {misclassification_rate:.4f}, Accuracy: {accuracy:.4f}")

# Steps 5 and 6: Train k-d tree models and compute misclassification rates for k values ranging from 2 to 5 (Manhattan distance)
print("\nUsing k-d tree (Manhattan distance):")
for k in range(2, 6):
    knn_tree = KNeighborsClassifier(n_neighbors=k, algorithm='kd_tree', metric='manhattan')
    knn_tree.fit(X_train_scaled, y_train)
    y_pred_tree = knn_tree.predict(X_test_scaled)
    misclassification_rate_tree = 1 - accuracy_score(y_test, y_pred_tree)
    accuracy_tree = accuracy_score(y_test, y_pred_tree)
    print(f"k = {k}, Misclassification rate: {misclassification_rate_tree:.4f}, Accuracy: {accuracy_tree:.4f}")