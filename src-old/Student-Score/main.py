import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("student-mat.csv", sep=";").apply(
    LabelEncoder().fit_transform
)  # Convert the whole DF in integer based values.

X = df.drop("G3", axis=1).values
y = df["G3"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Defining a model
model = Lasso()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Print the stats
print(f"Accuracy: {model.score(X_test, y_test)}")

# Plotting the data
plot_df = pd.DataFrame({"Actual": y_test.flatten(), "Predicted": y_pred.flatten()})

plot_df.head(25).plot(kind="bar", figsize=(16, 10))
plt.grid(which="major", linestyle="-", linewidth="0.5", color="green")
plt.grid(which="minor", linestyle=":", linewidth="0.5", color="black")
plt.show()
