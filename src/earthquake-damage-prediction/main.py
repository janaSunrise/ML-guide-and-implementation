import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Read and clean the data
train_data = pd.read_csv("train_values.csv")
train_labels = pd.read_csv("train_labels.csv")
test_data = pd.read_csv("test_values.csv")
test_labels = pd.read_csv("submission_format.csv")
