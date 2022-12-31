import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

df = pd.read_csv("./africa_recession.csv")

X = df.drop(["growthbucket"], axis=1).values
y = df["growthbucket"].values

# Plot the predicting data
"""
df["growthbucket"].value_counts()
# Visualize this count
sns.countplot(df["growthbucket"],label="Count")
print(X.shape)
print(y.shape)
"""

X = SelectKBest(f_classif, k=5).fit_transform(X, y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create the model
model = RandomForestClassifier()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Print stats
print(f"Accuracy: {accuracy_score(y_pred, y_test)}")
