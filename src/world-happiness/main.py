import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

dataframe = pd.read_csv("2015.csv")
df = dataframe.drop(["Region", "Happiness Rank"], axis=1)

X = df.drop("Happiness Score", axis=1).values
y = df["Happiness Score"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(f"Score: {model.score(X_test, y_test)}")
print(f"Classification report: {classification_report(y_test, y_pred)}")
print(f"Confusion matrix:\n{confusion_matrix(y_train, y_pred)}")
