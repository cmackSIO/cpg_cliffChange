import os
import laspy
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

def read_las(file_path):
    las_file = laspy.read(file_path)

    # Extract header information for scaling
    scale_x = las_file.header.scale[0]
    scale_y = las_file.header.scale[1]
    scale_z = las_file.header.scale[2]

    # Apply scale factors to x, y, and z
    x = las_file.x * scale_x
    y = las_file.y * scale_y
    z = las_file.z * scale_z
    intensity = las_file.intensity.astype(float)

    return pd.DataFrame({'x': x, 'y': y, 'z': z, 'intensity': intensity})

# Paths to manually cropped october and march surveys
janBeach = '/Volumes/group/LiDAR/LidarProcessing/changedetection_m3c2/m3c2_tools/training_sets/delmar/0109_beach.las'
janCliff = '/Volumes/group/LiDAR/LidarProcessing/changedetection_m3c2/m3c2_tools/training_sets/delmar/0109_cliff.las'
decBeach = '/Volumes/group/LiDAR/LidarProcessing/changedetection_m3c2/m3c2_tools/training_sets/delmar/1212_beach.las'
decCliff = '/Volumes/group/LiDAR/LidarProcessing/changedetection_m3c2/m3c2_tools/training_sets/delmar/1212_cliff.las'

# Read in the training clouds
jBeach = read_las(janBeach)
print(jBeach.head(5))
jCliff = read_las(janCliff)
dBeach = read_las(decBeach)
dCliff = read_las(decCliff)

# Add indicator columns
# 1 is beach, 0 is cliffs
jBeach['label'] = 1
dBeach['label'] = 1
jCliff['label'] = 0
dCliff['label'] = 0

# Combine into one big dataframe
# Concatenate DataFrames
xyzTrain = pd.concat([jBeach, jCliff, dBeach, dCliff], ignore_index=True)

print(xyzTrain.head(5))

# Convert specific columns to float64
columns_to_convert = ['x', 'y', 'z', 'intensity']
xyzTrain[columns_to_convert] = xyzTrain[columns_to_convert].apply(pd.to_numeric, errors='coerce')

# Print information about NaN values
nan_counts = xyzTrain.isna().sum()
print("NaN counts per column:")
print(nan_counts)

# Check for NaN values and drop rows with NaN
if xyzTrain.isna().any().any():
    print("Warning: NaN values found. Removing rows with NaN.")
    xyzTrain = xyzTrain.dropna()

# Print the number of samples after NaN removal
print(f"Number of samples after NaN removal: {len(xyzTrain)}")

# Check if there are enough samples for train-test split
if len(xyzTrain) == 0:
    print("Error: Not enough samples after NaN removal. Exiting.")
    exit()

# Create features and labels
x = xyzTrain.iloc[:, :4]
y = xyzTrain.iloc[:, 4]

# Train-test split on 80% of total data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=5)

# Create and train the Random Forest Classifier
winterModel = RandomForestClassifier(n_estimators=15, random_state=42)
winterModel.fit(x_train, y_train)

# Evaluate the model on the test set
y_pred = winterModel.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Model Accuracy on Test Set: {accuracy}')

# Save the trained model
model_path = "/Volumes/group/LiDAR/LidarProcessing/changedetection_m3c2/m3c2_tools/training_sets/delmar/model.joblib"
joblib.dump(winterModel, model_path)
print(f'Model saved to: {model_path}')