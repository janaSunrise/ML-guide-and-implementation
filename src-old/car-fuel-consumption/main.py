import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("measurements.csv", decimal=",")

df = df.drop(["specials", "refill liters", "refill gas"], axis=1).replace("?", np.nan)

df = df.fillna(df.mean().to_dict()).apply(LabelEncoder().fit_transform)

X = df.drop("consume", axis=1).values
y = df["consume"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create the model
model = RandomForestRegressor()

model.fit(X_train, y_train)

# Print stats
print(f"Accuracy: {model.score(X_test, y_test)}")
