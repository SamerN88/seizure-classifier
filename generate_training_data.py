# Import tools
import os
import csv
import datetime

import mne
import numpy as np
from scipy.signal import welch

from util import Stopwatch, print_runtime_sec

mne.set_log_level('ERROR')


########################################################################################################################
# SETUP PROJECT PATHS AND DIRECTORIES
########################################################################################################################


# Define project's paths
ROOT_DIR_PATH = os.path.abspath('.')
DATA_DIR_PATH = os.path.join(ROOT_DIR_PATH, 'data')
RECORDS_DIR_PATH = os.path.join(DATA_DIR_PATH, 'physionet-records')
TRAINING_DATA_DIR_PATH = os.path.join(DATA_DIR_PATH, 'training-data')
SEIZURE_CSV_DIR_PATH = os.path.join(TRAINING_DATA_DIR_PATH, 'seizure')
CONTROL_CSV_DIR_PATH = os.path.join(TRAINING_DATA_DIR_PATH, 'control')

# Create directories if they don't exist
os.makedirs(DATA_DIR_PATH, exist_ok=True)
os.makedirs(RECORDS_DIR_PATH, exist_ok=True)
os.makedirs(TRAINING_DATA_DIR_PATH, exist_ok=True)
os.makedirs(SEIZURE_CSV_DIR_PATH, exist_ok=True)
os.makedirs(CONTROL_CSV_DIR_PATH, exist_ok=True)


# Function for obtaining full path to a record (note that record_name usually will have format "chb01/chb01_01.edf", or sometimes just "chb01/<FILENAME>")
def fp(record_name):
    return os.path.join(RECORDS_DIR_PATH, record_name)


# Function to run wget Terminal command that gets resources from PhysioNet and saved them locally (in Google Drive)
def run_wget(local_dir_path, remote_path, *, verbose=False):
    local_dir_path = local_dir_path.replace(' ', r'\ ')
    output_config = '' if verbose else '> /dev/null 2>&1'
    os.system(f'wget -r -N -c -np -P {local_dir_path} --no-host-directories --cut-dirs=4 https://physionet.org/files/chbmit/1.0.0/{remote_path} {output_config}')


########################################################################################################################
# GET RECORD NAMES
########################################################################################################################


# Download initial files from PhysioNet's CHB-MIT Scalp EEG Database
run_wget(RECORDS_DIR_PATH, 'RECORDS')
run_wget(RECORDS_DIR_PATH, 'RECORDS-WITH-SEIZURES')

# Get list of record names
with open(fp('RECORDS'), 'r') as f:
    RECORDS = sorted(set(line.strip() for line in f.readlines() if line.strip() != ''))

# Get set of names of records that have seizures
with open(fp('RECORDS-WITH-SEIZURES'), 'r') as f:
    RECORDS_WITH_SEIZURES = set(line.strip() for line in f.readlines() if line.strip() != '')

print(f'Total records: {len(RECORDS)}')
print(f'Records with seizures: {len(RECORDS_WITH_SEIZURES)}')
print()


########################################################################################################################
# FEATURE EXTRACTION FUNCTIONS FOR EEG DATA (BOTH FREQUENCY-DOMAIN AND TIME-DOMAIN FEATURES)
########################################################################################################################


# Define frequency bands
FREQ_BANDS = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 45)
}


def compute_band_power(eeg_data, sfreq, band, window_sec=4):
    low, high = band
    nperseg = int(window_sec * sfreq)
    freqs, psd = welch(eeg_data, sfreq, nperseg=nperseg, axis=1)
    band_power = np.sum(psd[:, (freqs >= low) & (freqs <= high)], axis=1)
    return band_power


# Extracts frequency-domain features from EEG data
def freq_domain_features(eeg_data, sfreq):
    ordered_band_names = ['delta', 'theta', 'alpha', 'beta', 'gamma']
    band_powers = [compute_band_power(eeg_data, sfreq, FREQ_BANDS[band]) for band in ordered_band_names]
    return band_powers


# Extracts time-domain features from EEG data
def time_domain_features(eeg_data):
    mean = np.mean(eeg_data, axis=1)
    std = np.std(eeg_data, axis=1)
    variance = np.var(eeg_data, axis=1)
    skewness = np.apply_along_axis(lambda x: np.mean((x - np.mean(x))**3) / (np.std(x)**3), 1, eeg_data)
    kurtosis = np.apply_along_axis(lambda x: np.mean((x - np.mean(x))**4) / (np.std(x)**4), 1, eeg_data)
    return mean, std, variance, skewness, kurtosis


# Extracts all desired features from EEG data, consolidated in a single feature vector
def extract_features(eeg_data, sfreq):
    # Time-domain features
    mean, std, variance, skewness, kurtosis = time_domain_features(eeg_data)

    # Frequency-domain features
    delta_bp, theta_bp, alpha_bp, beta_bp, gamma_bp = freq_domain_features(eeg_data, sfreq)

    # Combine into a single feature array
    features = np.concatenate([
        mean, std, variance, skewness, kurtosis,
        delta_bp, theta_bp, alpha_bp, beta_bp, gamma_bp
    ], axis=0)

    return features  # since we have 23 channels and 10 features, the feature vector will have shape (230,)


########################################################################################################################
# DOWNLOAD SUMMARY FILES AND GET START/END TIMES OF ALL SEIZURES
########################################################################################################################


# Download all summary files
patients = sorted(set(record_name.split('/')[0] for record_name in RECORDS))  # not necessary to sort, just for orderliness
for i, patient_dir in enumerate(patients, 1):
    print(f'Downloading summary {i}/{len(patients)}')

    # Get summary filepath from patient identifier, then download summary file to Google Drive
    summary_fp = f'{patient_dir}/{patient_dir}-summary.txt'
    run_wget(fp(patient_dir), summary_fp)
print()

# Get start and end times of all seizures
patients_with_seizures = set(record_name.split('/')[0] for record_name in RECORDS_WITH_SEIZURES)
seizure_start_end_times = {}
for patient_dir in sorted(patients_with_seizures):  # not necessary to sort, just for orderliness
    # Get summary filepath from patient identifier
    summary_fp = f'{patient_dir}/{patient_dir}-summary.txt'

    with open(fp(summary_fp), 'r') as f:
        lines = f.readlines()

    # Parse summary file to get start and end times of seizures
    for i, line in enumerate(lines):
        if not line.startswith('Number of Seizures in File: '):
            continue

        num_seizures = int(line.split(': ')[1])
        if num_seizures == 0:
            continue

        # Parse record name and times
        record_name = f'{patient_dir}/{lines[i - 3].strip().split(": ")[1]}'
        start_time = float(lines[i + 1].split(': ')[1].split()[0])
        end_time = float(lines[i + 2].split(': ')[1].split()[0])

        seizure_start_end_times[record_name] = (start_time, end_time)


########################################################################################################################
# CORE FUNCTIONS FOR GENERATING TRAINING DATA
########################################################################################################################


# Segments EEG time series data into segments of 4 seconds each
def segment_data(eeg_data, sfreq, window_sec=4, drop_partial=True):
    num_time_slices = eeg_data.shape[1]
    window_size = int(window_sec * sfreq)
    segments = [eeg_data[:, i: i + window_size] for i in range(0, num_time_slices, window_size)]

    if len(segments) == 0:
        return []

    if drop_partial and segments[-1].shape[1] != window_size:
        segments.pop()

    return segments


# Takes a record name and returns two lists of training examples obtained from that record, one for seizures and one
# for non-seizures (control group). Each training example has a feature vector AND a label. A record produces multiple
# training examples, each example accounting for 4 seconds of EEG data.
def get_training_data_from_record(record_name):
    # Load EEG recording file
    file_path = fp(record_name)
    raw = mne.io.read_raw_edf(file_path, preload=True)
    data, times = raw[:]

    # Sampling frequency (num time slices per second)
    sfreq = raw.info['sfreq']

    if record_name in seizure_start_end_times.keys():
        # Get start and end indices (time slices) for seizure portion of data
        szr_start, szr_end = seizure_start_end_times[record_name]
        szr_start_idx = int(szr_start * sfreq)
        szr_end_idx = int(szr_end * sfreq)

        # Separate seizure data from non-seizure (control) data
        szr_data = data[:, szr_start_idx: szr_end_idx]
        ctl_data = np.concatenate((data[:, :szr_start_idx], data[:, szr_end_idx:]), axis=1)
    else:
        # If record does not contain seizure, use the whole record as control data
        szr_data = None
        ctl_data = data

    # Segment data
    if szr_data is None:
        szr_segments = []
    else:
        szr_segments = segment_data(szr_data, sfreq)

    ctl_segments = segment_data(ctl_data, sfreq)

    # Extract feature vectors for seizure segments, and assign label 1 (seizure present)
    szr_data = []
    for seg in szr_segments:
        szr_features = extract_features(seg, sfreq)
        szr_data.append((szr_features, 1))  # 1 represents seizure

    # Extract feature vectors for non-seizure segments, and assign label 0 (seizure absent)
    ctl_data = []
    for seg in ctl_segments:
        ctl_features = extract_features(seg, sfreq)
        ctl_data.append((ctl_features, 0))  # 0 represents non-seizure

    return szr_data, ctl_data


########################################################################################################################
# UTILITIES FOR GENERATING TRAINING DATA
########################################################################################################################


# Define some tools to be used when generating training data

# Define CSV columns for features and label
BASE_FEATURE_NAMES = [
    'mean',
    'std',
    'variance',
    'skewness',
    'kurtosis',
    'delta_BP',
    'theta_BP',
    'alpha_BP',
    'beta_BP',
    'gamma_BP',
]
num_channels = 23  # the EEG data was collected using 23 electrodes (channels) on the scalp
CSV_COLUMN_NAMES = []
for feature_name in BASE_FEATURE_NAMES:
    for channel_num in range(1, num_channels + 1):
        CSV_COLUMN_NAMES.append(f'{feature_name}_c{channel_num}')
CSV_COLUMN_NAMES.append('seizure')  # label


def get_csv_fp(csv_num, is_szr):
    dir_path = SEIZURE_CSV_DIR_PATH if is_szr else CONTROL_CSV_DIR_PATH
    example_type = "seizure" if is_szr else "control"
    return os.path.join(dir_path, f'{example_type}_examples_{csv_num}.csv')


# Create CSV file and write the headers (column names)
def initialize_csv(csv_fp):
    # To avoid accidentally overwriting data, we do nothing and return False if the specified CSV file already exists
    if os.path.exists(csv_fp):
        return False

    with open(csv_fp, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(CSV_COLUMN_NAMES)

    # Return True to indicate that a new CSV file was initialized
    return True


def append_data_to_csv(csv_fp, data):
    with open(csv_fp, 'a', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        for features, label in data:
            # Flatten features and append the label
            row = list(features) + [label]
            csv_writer.writerow(row)


def exceeds_max_file_size(filepath, max_file_size_bytes):
    # If file doesn't exist, return False
    return os.path.exists(filepath) and os.path.getsize(filepath) >= max_file_size_bytes


########################################################################################################################
# GENERATE TRAINING DATA AND SAVE TO CSV FILES
########################################################################################################################

print('\n' * 3)
print('=' * 100)
print('GENERATE TRAINING DATA')
print('=' * 100)
print('\n')

# Generate training examples from each record and save to CSV file; we start a new CSV file once it gets too large

# Once a CSV reaches this file size, we start a new CSV
MAX_FILE_SIZE_BYTES = 2 * (1024 * 1024 * 1024)  # 2 GiB  # TODO FOR USER: modify as desired (in bytes)

# Initialize CSV filepaths
szr_csv_num = 1
ctl_csv_num = 1
szr_csv_fp = get_csv_fp(szr_csv_num, True)
ctl_csv_fp = get_csv_fp(ctl_csv_num, False)

# Dynamically figure out which CSV file we should start writing to, based on the chosen MAX_FILE_SIZE_BYTES. On the
# first run, the CSV number will just be 1, but if it's not the first run there may already be max-size CSV files.
while exceeds_max_file_size(szr_csv_fp, MAX_FILE_SIZE_BYTES):
    szr_csv_num += 1
    szr_csv_fp = get_csv_fp(szr_csv_num, True)
while exceeds_max_file_size(ctl_csv_fp, MAX_FILE_SIZE_BYTES):
    ctl_csv_num += 1
    ctl_csv_fp = get_csv_fp(ctl_csv_num, False)

# Figure out which record to start from, in case this is not the first run (e.g. script was previously interrupted)
last_success_fp = 'LAST_SUCCESS.txt'
if not os.path.exists(last_success_fp):
    start_record = RECORDS[0]
else:
    with open(last_success_fp, 'r') as f:
        last_successful_record = f.read().strip().split('\n')[-1].strip()
    if RECORDS.index(last_successful_record) == len(RECORDS) - 1:
        print('*** All records have already been processed successfully. Exiting. ***')
        exit()
    start_record = RECORDS[RECORDS.index(last_successful_record) + 1]

# Again, if script was previously interrupted, we shouldn't re-process everything; only process remaining records
records_to_process = RECORDS[RECORDS.index(start_record):]

# Initialize CSVs if needed (the `initialize_csv` function ensures not to overwrite existing CSV files)
if not initialize_csv(szr_csv_fp):
    print(f'CSV file "{szr_csv_fp}" already exists; adding to it.')
if not initialize_csv(ctl_csv_fp):
    print(f'CSV file "{ctl_csv_fp}" already exists; adding to it.')
print()

# Iterate through records and write data to CSVs (both seizure and control data)
total_sw = Stopwatch()
total_sw.start()
print(f'Starting from record "{start_record}"{" (first record)" if start_record == RECORDS[0] else ""}')
print(f'Max CSV file size: {MAX_FILE_SIZE_BYTES} B')
print(f'\nStart time: {datetime.datetime.now(datetime.UTC)}\n---\n\n')
for i, record_name in enumerate(records_to_process, 1):
    print(f'PROCESSING RECORD {i}/{len(records_to_process)}')
    sw = Stopwatch()
    sw.start()

    # If needed, start new seizure CSV file
    if os.path.getsize(szr_csv_fp) >= MAX_FILE_SIZE_BYTES:
        szr_csv_num += 1
        szr_csv_fp = get_csv_fp(szr_csv_num, True)
        initialize_csv(szr_csv_fp)
        print('    ' + '=' * 75)
        print(f'    NEW "SEIZURE" CSV FILE: {szr_csv_fp}')
        print('    ' + '=' * 75)

    # If needed, start new control CSV file
    if os.path.getsize(ctl_csv_fp) >= MAX_FILE_SIZE_BYTES:
        ctl_csv_num += 1
        ctl_csv_fp = get_csv_fp(ctl_csv_num, False)
        initialize_csv(ctl_csv_fp)
        print('    ' + '=' * 75)
        print(f'    NEW "CONTROL" CSV FILE: {ctl_csv_fp}')
        print('    ' + '=' * 75)

    # Download record
    print(f'    Downloading record "{record_name}"... ', end='', flush=True)
    sw.lap()
    patient_dir = record_name.split('/')[0]
    run_wget(fp(patient_dir), record_name)
    print_runtime_sec(sw.lap())

    # Extract features
    print(f'    Extracting features... ', end='', flush=True)
    szr_training_data, ctl_training_data = get_training_data_from_record(record_name)
    print_runtime_sec(sw.lap())

    # Append training examples to respective CSV files
    print(f'    Writing to CSV files ({szr_csv_fp.split("/")[-1]}, {ctl_csv_fp.split("/")[-1]})... ', end='', flush=True)
    append_data_to_csv(szr_csv_fp, szr_training_data)
    append_data_to_csv(ctl_csv_fp, ctl_training_data)
    print_runtime_sec(sw.lap())

    # Log this record as the last successfully processed record
    with open(last_success_fp, 'w') as f:
        f.write(f'Last successfully processed record:\n{record_name}')

    print_runtime_sec(sw.total_elapsed())
    print()

print(f'\n---\nEnd time: {datetime.datetime.now(datetime.UTC)}')
print(f'Total runtime: {total_sw.total_elapsed():.3f} s')
