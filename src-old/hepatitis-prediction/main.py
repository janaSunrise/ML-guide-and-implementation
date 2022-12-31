import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

column_names = [
    "class",
    "age",
    "sex",
    "steroid",
    "antivirals",
    "fatigue",
    "malaise" "anorexia",
    "liver_big",
    "liver_firm",
    "spleen_palpable",
    "spiders",
    "ascites",
    "varices",
    "bilirubin",
    "alk_phosphate",
    "sgot",
    "albumin",
    "protime",
    "histology",
]

df = (
    pd.read_csv("hepatitis.csv", names=column_names)
    .replace("?", np.nan)
    .apply(pd.to_numeric)
    .drop_duplicates()
)

# Clean the data
df = df.fillna(df.mean().to_dict())
df["class"] = df["class"].astype("bool")

df = pd.get_dummies(df, drop_first=True)

df["bilirubin"] = np.abs(
    (df["bilirubin"] - df["bilirubin"].mean()) / df["bilirubin"].std()
)
df["albumin"] = np.abs((df["albumin"] - df["albumin"].mean()) / df["albumin"].std())

X = df.drop("class", axis=1).values
y = df["class"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create the model
knn = KNeighborsClassifier()

knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

# Print the stats
print(f"Accuracy of KNN (%): {accuracy_score(y_test, y_pred)}")
