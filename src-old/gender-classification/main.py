import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv("./Transformed Data Set - Sheet1.csv").apply(
    LabelEncoder().fit_transform
)

X = df.drop(["Gender"], axis=1)
y = df.Gender

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Defining the model
model = DecisionTreeClassifier(max_depth=4, min_samples_split=6, min_samples_leaf=5)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Printing the stats
print(f"Accuracy: {accuracy_score(y_pred, y_test)}")
