import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv("carbon_nanotubes.csv", decimal=",")

X = df.drop(["Chiral indice n", "Chiral indice m"], axis=1).values
y = df[["Chiral indice n", "Chiral indice m"]].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create the multiclassifier model
model = MultiOutputClassifier(DecisionTreeClassifier())

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Print score
print(f"Score: {model.score(X_test, y_test)}")
