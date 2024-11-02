import os

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler

from util import Stopwatch, print_runtime_sec


########################################################################################################################
# INITIAL SETUP
########################################################################################################################


sw = Stopwatch()
sw.start()

RANDOM_STATE = 61

# Paths to the CSV directories
SEIZURE_CSV_DIR = os.path.join('data', 'training-data', 'seizure')
CONTROL_CSV_DIR = os.path.join('data', 'training-data', 'control')

# Get all CSV filepaths (both seizure and control data)
seizure_csv_fps = sorted([os.path.join(SEIZURE_CSV_DIR, fn) for fn in os.listdir(SEIZURE_CSV_DIR) if fn.endswith('.csv')])
control_csv_fps = sorted([os.path.join(CONTROL_CSV_DIR, fn) for fn in os.listdir(CONTROL_CSV_DIR) if fn.endswith('.csv')])

# Get the column names
with open(seizure_csv_fps[0], 'r') as f:
    COLUMN_NAMES = f.readlines()[0].strip().split(',')


########################################################################################################################
# PREPARE DATA FOR TRAINING/TESTING
########################################################################################################################


# Helper function that takes a CSV filepath and returns a clean list of data rows
def get_rows_from_csv(csv_fp, cap=None):
    with open(csv_fp, 'r') as csv_file:
        lines = csv_file.readlines()

    # Iterate through lines of the CSV, convert to rows, and keep only rows that have the correct number of features
    rows = []
    for line in lines[1:]:
        # Turn str line into list of floats
        row = list(map(float, line.strip().split(',')))

        # Number of features must match number of columns
        if len(row) != len(COLUMN_NAMES):
            continue

        # Convert label to int, then add row to our data
        row[-1] = int(row[-1])
        rows.append(row)

        # Cap the number of examples as needed
        if cap is not None and len(rows) == cap:
            break

    return rows


# For now, we just use the first CSV file in each group

sw.lap()
print('Loading "seizure" data... ', end='', flush=True)
seizure_rows = get_rows_from_csv(seizure_csv_fps[0])
print_runtime_sec(sw.lap())

# The dataset is extremely imbalanced in favor of control data, so we balance it by truncating the control group.
# In order to not eliminate too much data diversity, we allow several times as much control data as seizure data.
CONTROL_RATIO = 10  # ratio of num control examples to num seizure examples

print('Loading "control" data... ', end='', flush=True)
control_rows = get_rows_from_csv(control_csv_fps[0], cap=CONTROL_RATIO*len(seizure_rows))
print_runtime_sec(sw.lap())

print('\nPreparing DataFrame for training... ', end='', flush=True)

# Create DataFrames
control_df = pd.DataFrame(control_rows, columns=COLUMN_NAMES)
seizure_df = pd.DataFrame(seizure_rows, columns=COLUMN_NAMES)

# Combine seizure and control data in one training dataset, and shuffle
data = pd.concat([seizure_df, control_df], ignore_index=True)
data = data.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

# Separate features and labels
X = data.drop(columns=['seizure'])
y = data['seizure']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

# Normalize data; fit the scaler on the train data, then transform both the train and test data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


########################################################################################################################
# TRAIN THE MODEL
########################################################################################################################


# Initialize the Random Forest classifier; we weight the classes according to the ratio control:seizure
seizure_classifier = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, class_weight={0: 1, 1: CONTROL_RATIO})

print_runtime_sec(sw.lap())  # runtime of preparing DataFrame for training
print(f'# train examples: {len(X_train)} [{sum(y_train == 1)} seizure, {sum(y_train == 0)} control]')
print(f'# test examples: {len(X_test)} [{sum(y_test == 1)} seizure, {sum(y_test == 0)} control]')

# Fit (train) the model
sw.lap()
print(f'\nTraining model ({type(seizure_classifier).__name__})... ', end='', flush=True)
seizure_classifier.fit(X_train, y_train)
print_runtime_sec(sw.lap())


########################################################################################################################
# TEST THE MODEL
########################################################################################################################


# Predict on the test set
y_pred = seizure_classifier.predict(X_test)

# Print evaluation metrics
print('\nPERFORMANCE ON TEST DATASET:\n')
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred), sep='')
print("Classification Report:\n", classification_report(y_test, y_pred), sep='')

# Test model on a large amount of unseen control data (unfortunately seizure data is too limited for this)
print('\n\n')
print('=' * 75)
print('TEST ON LARGE AMOUNT OF UNSEEN CONTROL DATA'.center(75) + '\n')
print('    Note: While control data was abundant, seizure data was limited, so \n'
      '          we used it all for training and initial testing. Consequently, \n'
      '          we cannot test on a large amount of unseen seizure data.')
print('=' * 75)
print()

sw.lap()
print('Loading unseen "control" data... ', end='', flush=True)
new_control_rows = get_rows_from_csv('data/training-data/control/control_examples_3.csv')  # TODO FOR USER: modify the path(s) as desired
new_control_df = pd.DataFrame(new_control_rows, columns=COLUMN_NAMES)
print_runtime_sec(sw.lap())

new_X = new_control_df.drop(columns=['seizure'])
new_y = new_control_df['seizure']
print(f'# unseen control examples: {len(new_control_df)}')

# Normalize (only transform, do not fit the scaler as it was already fit on the training data)
new_X = scaler.transform(new_X)

# Predict
new_y_pred = seizure_classifier.predict(new_X)

# Evaluation metrics
print("\nAccuracy:", accuracy_score(new_y, new_y_pred))
print('_' * 75)

# Save model
model_fn = 'seizure_classifier.joblib'
joblib.dump(seizure_classifier, model_fn)
print(f'\n\nModel saved at "{model_fn}". Load model using joblib.load("{model_fn}")')


# PROGRAM OUTPUT:
'''
Loading "seizure" data... [0.133 s]
Loading "control" data... [3.045 s]

Preparing DataFrame for training... [0.124 s]
# train examples: 7840 [714 seizure, 7126 control]
# test examples: 1961 [177 seizure, 1784 control]

Training model (RandomForestClassifier)... [3.588 s]

PERFORMANCE ON TEST DATASET:

Accuracy: 0.9969403365629781
Confusion Matrix:
[[1784    0]
 [   6  171]]
Classification Report:
              precision    recall  f1-score   support

           0       1.00      1.00      1.00      1784
           1       1.00      0.97      0.98       177

    accuracy                           1.00      1961
   macro avg       1.00      0.98      0.99      1961
weighted avg       1.00      1.00      1.00      1961




===========================================================================
                TEST ON LARGE AMOUNT OF UNSEEN CONTROL DATA                

    Note: While control data was abundant, seizure data was limited, so 
          we used it all for training and initial testing. Consequently, 
          we cannot test on a large amount of unseen seizure data.
===========================================================================

Loading unseen "control" data... [7.133 s]
# unseen control examples: 43015

Accuracy: 0.9576891781936534
___________________________________________________________________________


Model saved at "seizure_classifier.joblib". Load model using joblib.load("seizure_classifier.joblib")
'''
