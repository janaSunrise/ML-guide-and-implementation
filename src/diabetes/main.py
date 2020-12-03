import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV

# Get the dataset
df = pd.read_csv("diabetes.csv")

X = df.drop("diabetes", axis=1).values
y = df["diabetes"].values

# Split into sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define the Hyperparameters
model_params = {
    "n_estimators": [100, 10, 1]
}

# Define the model
model = RandomForestClassifier()

# Init the GridSearchCV
model_cv = GridSearchCV(model, param_grid=model_params)

# Fit the model
model_cv.fit(X_train, y_train)

# Print the best params
print(f"Best params: {model_cv.best_params_}")

# Predict
y_pred = model_cv.predict(X_test)

# Print the accuracy score
print(f"Accuracy: {model_cv.score(X_test, y_test)}")
