import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv("Data_train_reduced.csv")\
    .drop(["Respondent.ID", "Product.ID"], axis=1)\
    .apply(LabelEncoder().fit_transform)

X = df.drop(["Instant.Liking"], axis=1).values
y = df["Instant.Liking"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the model
model = DecisionTreeClassifier()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Print stats
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"Report: {classification_report(y_test, y_pred)}")
