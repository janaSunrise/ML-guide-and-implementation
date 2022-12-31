import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

df = pd.read_csv("student-mat-pass-or-fail.csv")[
    ["G1", "G2", "G3", "studytime", "failures", "absences", "pass", "traveltime"]
]

X = df.drop("G3", axis=1).values
y = df["G3"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create model
model = GradientBoostingRegressor(n_estimators=500)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Print stats
print(f"Accuracy: {model.score(X_test, y_test)}")
