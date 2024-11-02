# Seizure Classifier

## Setup

Simply run `pip install -r requirements.txt`.

## Contents

This repository contains all the necessary code to download EEG brainwave data from [PhysioNet](https://physionet.org/content/chbmit/1.0.0/), extract features, and train and test a Random Forest Classifier (RFC) to detect seizure events.

- `generate_training_data.py` - For downloading records from the dataset, extracting features, and generating usable training data.
- `train.py` - For training and testing a model.
- `util.py` - Some simple useful helper functions.
- `seizure_classifier.joblib` - My trained and saved RFC that can be loaded in Python and used.

## How to Run

You can download the repository and run the code as-is in the following order:

1. Run `generate_training_data.py` *(runtime: roughly 15-20 hours on a modern Mac machine)*
2. Run `train.py` *(runtime: less than a minute)*

However, if you want to avoid very long runtimes, you can actually interrupt the running of `generate_training_data.py` (#1 above) whenever you want and just use the data generated up to that point.

If you run this code as-is, it will produce the exact same model that is saved in `seizure_classifier.joblib`.

I've written a detailed [tutorial](https://docs.google.com/document/d/1n_W_VRPUNBavfys3Xo-w-1Bh0iAR_pWyawtw2pJXrTQ/edit?usp=sharing) that explains the rationale for this project and how to build it from scratch. It's in Google Doc format so you can easily navigate between sections.

### Important Note

The running of `generate_training_data.py` will create directories, subdirectories, and files to store and organize the generated training data. It will also create a file named `LAST_SUCCESS.txt` which is used to keep track of where the program left off if it gets interrupted. <ins>**_Do not delete this file_**</ins> unless you want to restart data generation from scratch, but then you’ll also have to delete all the data that has already been generated or else you’ll get duplicate data.

## References

Guttag, J. (2010). CHB-MIT Scalp EEG Database (version 1.0.0). _PhysioNet_. https://doi.org/10.13026/C2K01R.

<br>

---

Samer N. Najjar &nbsp;&nbsp; | &nbsp;&nbsp; s.najjar612@gmail.com &nbsp;&nbsp; | &nbsp;&nbsp; [Tutorial (Google Doc)](https://docs.google.com/document/d/1n_W_VRPUNBavfys3Xo-w-1Bh0iAR_pWyawtw2pJXrTQ/edit?usp=sharing)
