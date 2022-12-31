import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

column_names = [
    "symboling",
    "normalized_losses",
    "make",
    "fuel_type",
    "aspiration",
    "num_of_doors",
    "body_style",
    "drive_wheels",
    "engine_location",
    "wheel_base",
    "length",
    "width",
    "height",
    "curb_weight",
    "engine_type",
    "num_of_cylinders",
    "engine_size",
    "fuel_system",
    "bore",
    "stroke",
    "compression_ratio",
    "horsepower",
    "peak_rpm",
    "city_mpg",
    "highway_mpg",
    "price",
]

# Read and clean data

df = pd.read_csv("imports-85.csv", names=column_names)[
    [
        "body_style",
        "drive_wheels",
        "length",
        "width",
        "height",
        "curb_weight",
        "fuel_system",
        "stroke",
        "horsepower",
        "peak_rpm",
        "price",
    ]
]
df = df.replace("?", np.nan)
df = df.apply(
    lambda x: LabelEncoder().fit_transform(x)
    if x.name in ["body_style", "drive_wheels", "fuel_system"]
    else x
)

df = df.fillna(df.mean().to_dict())

# Split into sets
X = df.drop("price", axis=1).values
y = df["price"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create the model
model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Print stats
print(f"Accuracy: {model.score(y_test, y_pred)}")
