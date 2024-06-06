import pandas as pd
import joblib
import hashlib
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
import logging

# Setting up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Function to hash a string using SHA-256
def hash_string(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

# Function to scan data for potential issues and clean it
def scan_and_clean_data(df, dataset_name):
    logging.info("Scanning for malware...")
    alerts = []
    for col in df.columns:
        if df[col].dtype == 'object':
            numeric_column = pd.to_numeric(df[col], errors='coerce')
            if numeric_column.isnull().any():
                median_value = numeric_column.median()
                for index, (original_value, is_nan) in enumerate(zip(df[col], numeric_column.isnull())):
                    if is_nan:
                        df.at[index, col] = median_value
                        alert_info = {
                            'ID': index,
                            'HashValue': hash_string(str(original_value)),
                            'MachineName': col,
                            'Alert': 'Malware detected',
                            'RiskClassification': '2'
                        }
                        alerts.append(alert_info)
                        logging.info(f"Malware detected in {dataset_name}, from machine: {col} at index {index}: {original_value}")

    if alerts:
        pd.DataFrame(alerts).to_csv('alert.csv', index=False)
    return df

# Function to drop duplicates and unnecessary columns, and clean data
def dropping(df, dataset_name):
    df = df.drop_duplicates()
    df = df.dropna()
    if 'time_stamp' in df.columns:
        df = df.drop('time_stamp', axis=1)
    if 'ID' in df.columns:
        df = df.drop('ID', axis=1)
    df = scan_and_clean_data(df, dataset_name)
    logging.info("Data cleaning complete.")
    return df

# Main function to train a Random Forest model with k-fold cross-validation
def train_random_forest():
    logging.info("Machine learning phase started...")
    logging.info("Data loading...")
    
    # Load datasets
    df_normal = pd.read_csv('clean_dataset11.csv')
    df_extreme = pd.read_csv('Extreme_Values.csv')
    df_negative = pd.read_csv('Negative_Temp.csv')
    df_zeros = pd.read_csv('Zero_Values.csv')
    
    logging.info("Data cleaning...")
    
    # Clean datasets
    df_normal = dropping(df_normal, "Normal Dataset")
    df_extreme = dropping(df_extreme, "Extreme Dataset")
    df_negative = dropping(df_negative, "Negative Dataset")
    df_zeros = dropping(df_zeros, "Zero Dataset")

    # Label abnormal datasets
    df_extreme["label"] = "Anomaly - IoT / Machine input modification"
    df_negative["label"] = "Anomaly - Unauthorized access to server data"
    df_zeros["label"] = "Anomaly - Physical sensor tampering"

    # Combine normal and abnormal datasets
    df_abnormal = pd.concat([df_extreme, df_negative, df_zeros])
    df_normal["label"] = "Normal"
    df_combined = pd.concat([df_normal, df_abnormal])

    features = df_combined.columns.difference(["label"])

    logging.info("Data labeling completed...")
    
    # Standardize the features
    scaler = StandardScaler()
    df_combined[features] = scaler.fit_transform(df_combined[features])
    
    X = df_combined[features]
    y = df_combined["label"]

    logging.info("Building Random Forest model...")
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=60, random_state=40)

    logging.info("Performing k-fold cross-validation...")
    
    # Set up k-fold cross-validation
    kfold = KFold(n_splits=10, shuffle=True, random_state=40)
    start_time = time.time()
    cv_results = cross_val_score(rf_model, X, y, cv=kfold, scoring='accuracy')
    end_time = time.time()
    
    # Log cross-validation results
    logging.info(f"Cross-validation scores: {cv_results}")
    logging.info(f"Mean accuracy: {cv_results.mean():.2f} (+/- {cv_results.std():.2f})")
    logging.info(f"K-fold cross-validation took {end_time - start_time:.2f} seconds")

    # Plot cross-validation scores
    plt.figure()
    plt.plot(cv_results, marker='o', linestyle='--', color='b')
    plt.title('Cross-Validation Scores')
    plt.xlabel('Fold')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.savefig('cross_validation_scores.png')
    plt.show()

    logging.info("Training Random Forest model on the entire dataset...")
    
    # Train the model on the entire dataset
    rf_model.fit(X, y)

    logging.info("Saving the trained model and features...")
    
    # Save the model and feature names
    joblib.dump(rf_model, 'random_forest_model.pkl')
    joblib.dump(features.tolist(), 'features.pkl')  # Save the feature names

    logging.info("Random Forest model and features are saved.")
    
    logging.info("Generating predictions and confusion matrix...")
    
    # Generate predictions and confusion matrix
    y_pred = rf_model.predict(X)
    conf_matrix = confusion_matrix(y, y_pred)
    
    logging.info("Generating and saving heatmap...")
    
    # Plot and save the confusion matrix heatmap
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=rf_model.classes_, yticklabels=rf_model.classes_)
    plt.title("Confusion Matrix Heatmap")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig('confusion_matrix_heatmap.png')
    plt.show()

    logging.info("Heatmap saved as confusion_matrix_heatmap.png")
    logging.info("Machine learning phase completed.")

# Execute the training function
train_random_forest()
