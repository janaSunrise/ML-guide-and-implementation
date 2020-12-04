import pandas as pd

from matplotlib import pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston

X, y = load_boston(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)

reg_params = {
    "n_estimators": [100, 10, 1],
}

# Create the model
reg = RandomForestRegressor()

# Init the GridSearchCV
reg_cv = GridSearchCV(reg, param_grid=reg_params)

reg_cv.fit(X_train, y_train)

y_pred = reg_cv.predict(X_test)

print(f"Best params: {reg_cv.best_params_}")
print(f"Accuracy: {reg_cv.score(X_test, y_test)}")

# Plot Test vs Pred
plot_df = pd.DataFrame({
    'Actual': y_test.flatten(),
    'Predicted': y_pred.flatten()
})

plot_df.head(25).plot(kind='bar', figsize=(16, 10))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()
