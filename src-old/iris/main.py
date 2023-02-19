from sklearn.datasets import load_iris
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

X, y = load_iris(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

model = LinearDiscriminantAnalysis()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(f"Model score: {model.score(X_test, y_test)}")
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
