from sklearn.datasets import load_breast_cancer
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

X, y = load_breast_cancer(return_X_y=True)

classes = ["Malignant", "Benign"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

model = SVC(kernel="linear", gamma="auto")

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(f"Accuracy: {accuracy_score(y_pred, y_test)}")
print(f"Classification report: {classification_report(y_test, y_pred)}")
print(f"Confusion Matrix design:\n{confusion_matrix(y_test, y_pred)}")
