import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

df = pd.read_csv("real-estate.csv")\
       .drop(["No", "X5 latitude", "X6 longitude"], axis=1)

X = df.drop("Y house price of unit area", axis=1).values
y = df["Y house price of unit area"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create model
model = RandomForestRegressor()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Print stats
print(f"Accuracy: {model.score(X_test, y_test)}")

# Plot Test vs Pred
plot_df = pd.DataFrame({
    'Actual': y_test.flatten(),
    'Predicted': y_pred.flatten()
})

plot_df.head(25).plot(kind='bar', figsize=(16, 10))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()
