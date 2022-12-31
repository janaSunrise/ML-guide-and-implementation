import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.model_selection import train_test_split

df = (
    pd.read_csv("cervical_cancer.csv")
    .replace("?", np.nan)
    .apply(pd.to_numeric)
    .drop(
        columns=["STDs: Time since first diagnosis", "STDs: Time since last diagnosis"]
    )
    .drop_duplicates()
    .rename(columns={"Biopsy": "Cancer"})
)
df = df.fillna(df.mean().to_dict())

X = df.drop("Cancer", axis=1).values
y = df["Cancer"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Create the model
model = RandomForestClassifier(n_estimators=500)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Print the stats
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"Classification report: {classification_report(y_test, y_pred)}")
print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
