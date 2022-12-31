import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("weight-height.csv")
encoder = LabelEncoder()

# Split into sets
df["Gender"] = encoder.fit_transform(df["Gender"])

X = df.drop("Weight", axis=1).values
y = df["Weight"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create the model
model = LinearRegression()

model.fit(X_train, y_train)

print(f"Accuracy: {model.score(X_test, y_test)}")
