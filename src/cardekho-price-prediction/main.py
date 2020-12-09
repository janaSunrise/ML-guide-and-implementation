import re

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor


def unit_trim(string: str) -> str:
    return ''.join(re.findall(r"^[0-9]+\.?[0-9]+", str(string)))


# Read and clean the data
df = pd.read_csv("car-details.csv").drop(["name", "year", "torque"], axis=1).replace("?", np.nan).dropna()

for column in ["mileage", "engine", "max_power"]:
    df[column] = df[column].apply(unit_trim)

df = pd.get_dummies(df, columns=["fuel", "seller_type", "transmission", "owner"], drop_first=True)

X = df.drop("selling_price", axis=1).values
y = df["selling_price"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the model
model = DecisionTreeRegressor()

model.fit(X_test, y_test)

y_pred = model.predict(X_test)

# Show stats
print(f"Accuracy: {model.score(X_test, y_test)}")
