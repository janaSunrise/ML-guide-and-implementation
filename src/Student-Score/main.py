import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("student-mat.csv", sep=';')\
       .apply(LabelEncoder().fit_transform)  # Convert the whole DF in integer based values.

X = df.drop("G3", axis=1).values
y = df["G3"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Defining a model
model = Lasso()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Print the stats
print(f"Accuracy: {model.score(X_test, y_test)}")
