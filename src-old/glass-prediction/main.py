import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# Clean up data, and add column names
column_names = [
    "id",
    "refractive_index",
    "sodium",
    "magnesium",
    "aluminium",
    "silicon",
    "potassium",
    "calcium",
    "barium",
    "iron",
    "glass_type",
]

df = pd.read_csv("glass.csv", names=column_names).drop("id", axis=1)

X = df.drop("glass_type", axis=1).values
y = df["glass_type"].values

# Split into sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Define the model
model = GradientBoostingClassifier(n_estimators=500)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Print stats
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"Report: {classification_report(y_test, y_pred, zero_division=1)}")
