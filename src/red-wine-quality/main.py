import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("winequality-red.csv").dropna()

df["quality_value"] = df.quality.apply(lambda x: 'low' if x <= 5 else 'medium' if x <= 7 else 'high')

df = df.apply(LabelEncoder().fit_transform)

X = df.drop("quality_value", axis=1).values
y = df["quality_value"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create model
model = RandomForestClassifier()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Print stats
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"Report: {classification_report(y_test, y_pred)}")
