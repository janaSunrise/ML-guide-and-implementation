import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

columns_to_drop = ["Country", "Region", "Happiness Rank"]

df = pd.read_csv("2015.csv").drop(columns_to_drop, axis=1).dropna()

X = df.drop("Happiness Score", axis=1).values
y = df["Happiness Score"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the model
model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Print stats
print(f"Score: {model.score(X_test, y_test)}")
