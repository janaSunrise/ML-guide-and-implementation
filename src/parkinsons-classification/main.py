import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

df = pd.read_csv("parkinsons.csv").dropna().drop("name", axis=1)

X = df.drop("status", axis=1).values
y = df["status"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create model
model = GradientBoostingClassifier()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Print stats
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"Report: {classification_report(y_test, y_pred)}")
