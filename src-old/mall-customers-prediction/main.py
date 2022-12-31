import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeRegressor

df = pd.read_csv("Mall_Customers.csv")\
       .drop("CustomerID", axis=1)\
       .apply(LabelEncoder().fit_transform)  # Convert the whole DF into integer based values.

X = df.drop("Spending Score", axis=1).values
y = df["Spending Score"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create the model
model = DecisionTreeRegressor()

model.fit(X, y)

y_pred = model.predict(X_test)

# Print the score
print(f"Score: {model.score(X_test, y_test)}")

# Plot Test vs Pred
plot_df = pd.DataFrame({
    'Actual': y_test.flatten(),
    'Predicted': y_pred.flatten()
})

plot_df.head(25).plot(kind='bar', figsize=(16, 10))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()
