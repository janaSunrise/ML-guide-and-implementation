import pandas as pd

from matplotlib import pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston

# Read CSV using pandas
X, y = load_boston(return_X_y=True)

# Split into sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)

# Parameters for Ridge
rfg_params = {
    "n_estimators": [100, 10, 1],
}

# Create the model
rfg = RandomForestRegressor()

# Init the GridSearchCV
rfg_cv = GridSearchCV(rfg, param_grid=rfg_params)

# Fit the set
rfg_cv.fit(X_train, y_train)

# Print the best params
print(f"Best params: {rfg_cv.best_params_}")

# Predict
y_pred = rfg_cv.predict(X_test)

# Print the accuracy score
print(f"Accuracy: {rfg_cv.score(X_test, y_test)}")

# Do the plotting
plot_df = pd.DataFrame({
    'Actual': y_test.flatten(),
    'Predicted': y_pred.flatten()
})

plot_df.head(25).plot(kind='bar', figsize=(16, 10))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()
