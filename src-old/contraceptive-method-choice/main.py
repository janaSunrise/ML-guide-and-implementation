import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

column_names = [
    "wife_age",
    "wife_education",
    "husband_education",
    "number_children",
    "wife_religion",
    "is_wife_working",
    "husband_occupation",
    "standard_living_index",
    "media_exposure",
    "contraceptive_method",
]

df = pd.read_csv("cmc.csv", names=column_names)

X = df.drop("contraceptive_method", axis=1).values
y = df["contraceptive_method"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Make the model
model = ExtraTreesClassifier(criterion="entropy")

model.fit(X, y)

y_pred = model.predict(X_test)

# Print the stats
print(f"Score: {accuracy_score(y_test, y_pred)}")
print(f"Report: {classification_report(y_test, y_pred, zero_division=1)}")

# Plot Test vs Pred
plot_df = pd.DataFrame({"Actual": y_test.flatten(), "Predicted": y_pred.flatten()})

plot_df.head(25).plot(kind="bar", figsize=(16, 10))
plt.grid(which="major", linestyle="-", linewidth="0.5", color="green")
plt.grid(which="minor", linestyle=":", linewidth="0.5", color="black")
plt.show()
