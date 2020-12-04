import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV

# Get the dataset
df = pd.read_csv("diabetes.csv")

X = df.drop("diabetes", axis=1).values
y = df["diabetes"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model_params = {}

model = GradientBoostingClassifier()

model_cv = GridSearchCV(model, param_grid=model_params)

model_cv.fit(X_train, y_train)

y_pred = model_cv.predict(X_test)

print(f"Best params: {model_cv.best_params_}")
print(f"Accuracy: {model_cv.score(X_test, y_test)}")
