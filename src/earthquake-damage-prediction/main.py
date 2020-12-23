import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Read and clean the data
drop_columns = ["building_id"]

X_train = pd.read_csv("train_values.csv")\
    .drop(drop_columns, axis=1)\
    .apply(LabelEncoder().fit_transform)\
    .values
y_train = pd.read_csv("train_labels.csv")\
    .drop(drop_columns, axis=1)\
    .values

X_test = pd.read_csv("test_values.csv")\
    .drop(drop_columns, axis=1)\
    .apply(LabelEncoder().fit_transform)\
    .values
y_test = pd.read_csv("submission_format.csv")\
    .drop(drop_columns, axis=1)\
    .values

# Create the model
model = RandomForestClassifier()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Get scores
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
