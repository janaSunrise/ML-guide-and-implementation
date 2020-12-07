import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVR

df = pd.read_csv("forestfires.csv")\
       .drop(columns=['month', 'day'])\
       .drop_duplicates()

X = df.drop("area", axis=1).values
y = df["area"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearSVR()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

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
