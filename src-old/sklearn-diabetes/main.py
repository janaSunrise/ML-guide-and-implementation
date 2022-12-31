import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsRegressor

X, y = load_diabetes(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

params = {
    "n_neighbors": np.arange(1, 15),
}

model = KNeighborsRegressor()

model_cv = GridSearchCV(model, param_grid=params)

model_cv.fit(X_train, y_train)

print(f"Best params: {model_cv.best_params_}")
print(f"Score: {model_cv.score(X_test, y_test)}")
