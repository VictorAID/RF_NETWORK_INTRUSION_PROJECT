from data_loader import load_data
from preprocessing import normalize_features, reduce_dimensionality, label_anomalies
from model import train_test_split_data, train_model, evaluate_model
from save_load import save_model

# Load data
data = load_data("Network Intrusion.csv")

# Normalize features
features = data.drop('class', axis=1).columns
data_scaled, scaler = normalize_features(data, features)

# Dimensionality reduction
X_pca, pca = reduce_dimensionality(data_scaled)

# Label anomalies
data['type'] = label_anomalies(X_pca)

# Features and labels
X = data.drop(['class', 'type'], axis=1).values
y = data['type'].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split_data(X, y)

# Train model
model = train_model(X_train, y_train)

# Evaluate model
accuracy, report = evaluate_model(model, X_test, y_test)
print(f"Accuracy: {accuracy:.4f}")
print("Classification Report:")
print(report)

# Save the model
save_model(model, 'model.pkl')
