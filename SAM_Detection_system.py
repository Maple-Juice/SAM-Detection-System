import pandas as pd
import numpy as np
import hashlib
import joblib
import os
import time
import logging
import csv
from sklearn.preprocessing import StandardScaler

# Setting up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Dictionary to classify risk levels for different alerts
RISK_CLASSIFICATION = {
    "Malware detected": 2,
    "Anomaly - Physical sensor tampering": 3,
    "Anomaly - Unauthorized access to server data": 3,
    "Anomaly - IoT / Machine input modification": 3,
    "Data tampering detected": 2
}

# Function to load data from a CSV file
def load_data(file_path):
    df = pd.read_csv(file_path)
    df.set_index('ID', inplace=True)
    return df

# Function to load the hash file if it exists, otherwise create an empty DataFrame
def load_hash_file(hash_file_path):
    if os.path.exists(hash_file_path):
        return pd.read_csv(hash_file_path, index_col=[0, 1])
    else:
        return pd.DataFrame(columns=['Row', 'Column', 'Hash']).set_index(['Row', 'Column'])

# Function to generate a SHA-256 hash for a given data
def generate_hash(data):
    return hashlib.sha256(data.encode()).hexdigest()

# Function to update the hash file with new hash values
def update_hash_file(df_hashes, hash_file_path):
    df_hashes.reset_index(inplace=True)
    df_hashes.to_csv(hash_file_path, index=False)
    logging.info(f"Hash file updated at {hash_file_path}")

# Function to manage and compare hashes of the data, and detect anomalies
def manage_hashes(df, hash_file_path, alert_file_path):
    df_hashes = load_hash_file(hash_file_path)
    updated_hashes = pd.DataFrame(columns=['Hash'], index=pd.MultiIndex.from_product([df.index, df.columns], names=['Row', 'Column']))
    anomalies = []

    for index, row in df.iterrows():
        for column in df.columns:
            cell_value = str(row[column])
            cell_hash = generate_hash(cell_value)
            if (index, column) in df_hashes.index:
                if df_hashes.loc[(index, column), 'Hash'] != cell_hash:
                    anomaly_info = {
                        'ID': index,
                        'HashValue': cell_hash,
                        'MachineName': column,
                        'Alert': 'Data tampering detected',
                        'RiskClassification': RISK_CLASSIFICATION['Data tampering detected']
                    }
                    anomalies.append(anomaly_info)
                    logging.warning(f"Data tampering detected at row {index}, column {column}")
                    if index > 0:
                        # Replace the tampered cell with the value from the cell above
                        df.at[index, column] = df.at[index - 1, column]
                        logging.info(f"Replaced value at row {index}, column {column} with value from row {index - 1}")

            updated_hashes.loc[(index, column)] = cell_hash

    if anomalies:
        log_anomalies(anomalies, alert_file_path)
    update_hash_file(updated_hashes, hash_file_path)
    return df

# Function to load the trained model and feature names
def load_model_and_features(model_path, features_path):
    model = joblib.load(model_path)
    features = joblib.load(features_path)
    return model, features

# Function to handle missing values in the dataset
def handle_missing_values(df):
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    df.fillna(0, inplace=True)
    return df

# Function to scan the dataset for malware and replace detected values
def malware_scan_and_replace(df, alert_file_path):
    changed = False
    anomalies = []
    logging.info("Scanning for malware in the dataset...")
    for col in df.columns:
        if df[col].dtype == 'object':
            for i in range(len(df)):
                cell_value = str(df.iloc[i, df.columns.get_loc(col)])
                if not cell_value.replace('.', '', 1).isdigit():
                    cell_hash = generate_hash(cell_value)
                    if i == 0:
                        df.iloc[i, df.columns.get_loc(col)] = 0
                        changed = True
                        logging.info(f"Replaced malware in {col} at row {i} with 0 because it's the first row.")
                    else:
                        df.iloc[i, df.columns.get_loc(col)] = df.iloc[i-1, df.columns.get_loc(col)]
                        changed = True
                        logging.info(f"Replaced malware in {col} at row {i} with value from above cell.")
                    anomaly_info = {
                        'ID': i,
                        'HashValue': cell_hash,
                        'MachineName': col,
                        'Alert': 'Malware detected',
                        'RiskClassification': RISK_CLASSIFICATION['Malware detected']
                    }
                    anomalies.append(anomaly_info)
    if changed:
        logging.info(f"Dataset changes due to malware scanning.")
    if anomalies:
        log_anomalies(anomalies, alert_file_path)
    return df, changed

# Function to compare dataset features with model features and identify discrepancies
def compare_features(dataset_features, model_features):
    missing_features = [feature for feature in model_features if feature not in dataset_features]
    extra_features = [feature for feature in dataset_features if feature not in model_features]
    return missing_features, extra_features

# Dummy function to extract machine name from feature name (placeholder for actual logic)
def extract_machine_name(feature_name):
    return feature_name

# Function to sanitize detected anomalies by replacing values with previous row values
def sanitize_anomalies(df, anomalies):
    for anomaly in anomalies:
        id = anomaly['ID']
        if id > 0:
            df.iloc[id-1] = df.iloc[id]
    return df

# Function to detect anomalies using the trained model
def detect_anomalies(df, model, features):
    logging.info("Performing anomaly detection...")
    df_features = df[features]

    scaler = StandardScaler()
    df_features = pd.DataFrame(scaler.fit_transform(df_features), columns=features)

    predictions = model.predict(df_features)
    prediction_probs = model.predict_proba(df_features)
    
    df['Anomaly_Type'] = predictions
    df['Prediction_Confidence'] = prediction_probs.max(axis=1)

    high_confidence_anomalies = df[(df['Prediction_Confidence'] >= 0.7) & (df['Anomaly_Type'] != "Normal")]
    
    logging.info(f"Anomalies detected: {len(high_confidence_anomalies)}")
    
    anomalies = []
    for index, row in high_confidence_anomalies.iterrows():
        max_feature_index = np.argmax(prediction_probs[index])
        max_feature_name = features[max_feature_index]
        machine_name = extract_machine_name(max_feature_name)
        
        cell_value = str(row[max_feature_name])
        cell_hash = generate_hash(cell_value)
        
        alert_type = row['Anomaly_Type']
        risk_classification = RISK_CLASSIFICATION.get(alert_type, 3)  # Default to 3 if not found

        anomaly_info = {
            'ID': index,
            'HashValue': cell_hash,
            'MachineName': machine_name,
            'Alert': alert_type,
            'RiskClassification': risk_classification
        }
        anomalies.append(anomaly_info)
    
    logging.debug(f"High confidence anomalies: {anomalies}")

    return anomalies, df

# Function to log detected anomalies to a CSV file
def log_anomalies(anomalies, alert_file_path):
    header = ['ID', 'HashValue', 'MachineName', 'Alert', 'RiskClassification']
    file_exists = os.path.isfile(alert_file_path)
    with open(alert_file_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
        for anomaly in anomalies:
            logging.debug(f"Anomaly details: {anomaly}")
            id = anomaly.get('ID', '')
            hash_value = anomaly.get('HashValue', '')
            machine_name = anomaly.get('MachineName', '')
            alert = anomaly.get('Alert', '')
            risk_classification = anomaly.get('RiskClassification', '')
            if id and alert:
                logging.info(f"Logging anomaly: {id}, {hash_value}, {machine_name}, {alert}, {risk_classification}")
                writer.writerow([id, hash_value, machine_name, alert, risk_classification])

# Function to continuously run the anomaly detection system at specified intervals
def run_system_continuously(data_path, model_path, features_path, hash_file_path, alert_file_path, check_interval=10):
    while True:
        logging.info("Checking for data updates...")
        if os.path.exists(data_path):
            logging.info("Data Reloading...")
            dataset = load_data(data_path)
            logging.info("Handling missing values...")
            dataset = handle_missing_values(dataset)
            logging.info("Dropping 'time_stamp' column if it exists...")
            if 'time_stamp' in dataset.columns:
                dataset = dataset.drop(columns=['time_stamp'])
            logging.info("Malware Scanning...")
            dataset, malware_changed = malware_scan_and_replace(dataset, alert_file_path)
            
            logging.info("Loading Trained Model...")
            model, features = load_model_and_features(model_path, features_path)

            dataset_features = dataset.columns.tolist()
            missing_features, extra_features = compare_features(dataset_features, features)
            if missing_features:
                logging.warning(f"Missing features in the dataset: {missing_features}")
            if extra_features:
                logging.warning(f"Extra features in the dataset: {extra_features}")
            
            dataset = dataset[features]

            logging.info("Anomaly Detection (ML) initiated...")
            anomalies, df_with_predictions = detect_anomalies(dataset, model, features)
            logging.info(f"Anomalies detected for logging: {anomalies}")
            logging.info("Sanitizing detected anomalies...")
            dataset = sanitize_anomalies(df_with_predictions, anomalies)
            
            logging.info("Removing prediction columns before saving sanitized dataset...")
            if 'Prediction_Confidence' in dataset.columns:
                dataset = dataset.drop(columns=['Prediction_Confidence'])
            if 'Anomaly_Type' in dataset.columns:
                dataset = dataset.drop(columns=['Anomaly_Type'])
            
            logging.info("Saving sanitized dataset...")
            dataset.to_csv(data_path)
            
            logging.info("Hash comparison initiated...")
            dataset = manage_hashes(dataset, hash_file_path, alert_file_path)
            
            logging.info("Uploading Logs to alerting system...")
            log_anomalies(anomalies, alert_file_path)
            
            if malware_changed or anomalies:
                logging.info("Changes detected. Saving modified dataset...")
                dataset.to_csv(data_path)
            
            logging.info(f"System executed successfully at: {time.ctime()}")
        else:
            logging.warning(f"Data file not found: {data_path}")
        
        time.sleep(check_interval)

# Start the system to continuously check for anomalies
run_system_continuously("clean_dataset_test11.csv", "random_forest_model.pkl", "features.pkl", "hash.csv", "alert.csv")
