import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

df = pd.read_csv("SP500.csv").drop(["Date"], axis=1)

X = scaler.fit_transform(df.drop("LABEL", axis=1).values)
y = df["LABEL"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Create the model
model = RandomForestClassifier()

model.fit(X_test, y_test)

y_pred = model.predict(X_test)

# Print stats
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"Classification report: {classification_report(y_test, y_pred)}")
