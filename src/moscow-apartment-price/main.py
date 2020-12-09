import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("moscow_apartment_listings.csv")[[
    "price", "repair", "house_age", "closest_subway", "rooms", "footage", "floor", "hm"
]]

df = df.apply(
    lambda x: LabelEncoder().fit_transform(x) if x.name in ["hm"] else x
)

X = df.drop("price", axis=1).values
y = df["price"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model
model = RandomForestRegressor(n_estimators=200)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Print stats
print(f"Accuracy: {model.score(X_test, y_test)}")
