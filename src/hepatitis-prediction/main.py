import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

column_names = [
    "class", "age", "sex", "steroid", "antivirals", "fatigue", "malaise"
    "anorexia", "liver_big", "liver_firm", "spleen_palpable", "spiders",
    "ascites", "varices", "bilirubin", "alk_phosphate", "sgot", "albumin",
    "protime", "histology"
]

df = pd.read_csv("hepatitis.csv", names=column_names)\
    .replace("?", np.nan)\
    .apply(pd.to_numeric)\
    .drop_duplicates()

df = df.fillna(df.mean().to_dict())
df["class"] = df["class"].astype("bool")

df1 = pd.get_dummies(df, drop_first=True)


df1["bilirubin"] = np.abs((df1["bilirubin"]-df1["bilirubin"].mean())/(df1["bilirubin"].std()))
df1["albumin"] = np.abs((df1["albumin"]-df1["albumin"].mean())/(df1["albumin"].std()))

X = df1.drop(columns=["class"])
y = df1["class"].copy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
print("\n shape of train split: ")
print(X_train.shape, y_train.shape)
print("\n shape of train split: ")
print(X_test.shape, y_test.shape)

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
predictions = knn.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print("Accuracy of KNN (%): \n", accuracy)
