# SAM-Detection-System
Secure Anomaly monitoring and detection system - Securing Data Layer

Note: Some CSV Files May seem empty, but the program will generate its own values within the CSV files and utilize them for further processing.

in addition, 2 .png images will be generated once the random forest training is complete, providing more detailed insign about the model performance.


once the SAM-Detection system is executed, there might be file access / read / write clash between the user (you) and the program.

make sure to change the permission of the file with SYSTEM user to allow all. The following files permissions need to be changed (if not set automatically by the system):
- alert.csv
- hash.csv
- clean_dataset_test11.csv
- clean_dataset11.csv
