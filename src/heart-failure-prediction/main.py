import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from mlxtend.plotting import plot_confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

df = pd.read_csv("heart_failure_clinical_records_dataset.csv")

# Feature selection using a heatmap
plt.figure(figsize=(10, 10))
sns.heatmap(df.corr(), vmin=-1, cmap='coolwarm', annot=True)

Features = ['time', 'ejection_fraction', 'serum_creatinine', "smoking"]

X = df[Features].values
y = df["DEATH_EVENT"].values

# Spliting to test and train dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=2)

# initialising the model
model = GradientBoostingClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_pred, y_test)}")

# Ploting the confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure()
plot_confusion_matrix(cm, figsize=(12, 8), hide_ticks=True, cmap=plt.cm.Blues)
plt.title("Gradient Boosting Classifier")
plt.xticks(range(2), ["Heart Not Failed", "Heart Fail"], fontsize=16)
plt.yticks(range(2), ["Heart Not Failed", "Heart Fail"], fontsize=16)
plt.show()
