import pandas as pd
from matplotlib import pyplot as plt
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

train_df = pd.read_csv("train.csv")

X = train_df.drop(["price_range"], axis=1).values
y = train_df["price_range"].values

# Feature selection
fs = SelectKBest(score_func=f_classif, k=10)
X = fs.fit_transform(X, y)

# Spliting into train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Defining the model
model = KNeighborsClassifier(n_neighbors=9)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Printing the stats of the model
print(f"Accuracy: {accuracy_score(y_pred, y_test)}")
