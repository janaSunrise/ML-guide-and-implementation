import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Prepare datasets
train_ds = pd.read_csv("train.csv")
test_ds = pd.read_csv("test.csv")

df = pd.concat([train_ds, test_ds])
df = df.dropna()

X = df.drop("y", axis=1).values
y = df["x"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Make the model
model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Show stats
print(f"Accuracy: {model.score(X_test, y_test)}")

# Plot Test vs Pred
plot_df = pd.DataFrame({"Actual": y_test.flatten(), "Predicted": y_pred.flatten()})

plot_df.head(25).plot(kind="bar", figsize=(16, 10))
plt.grid(which="major", linestyle="-", linewidth="0.5", color="green")
plt.grid(which="minor", linestyle=":", linewidth="0.5", color="black")
plt.show()
