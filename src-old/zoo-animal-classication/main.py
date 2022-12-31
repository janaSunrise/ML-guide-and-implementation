import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, plot_confusion_matrix
from sklearn.model_selection import train_test_split

df = pd.read_csv("zoo.csv")
classes = pd.read_csv("class.csv")
classes = classes.drop(
    ["Number_Of_Animal_Species_In_Class", "Animal_Names"], axis=1
)

# Plotting the classes
train_dictionary = dict(df.class_type.value_counts())
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
          '#8c564b', '#e377c2']
names = list(classes.Class_Type)
train_values = train_dictionary.values()
plt.bar(names, train_values, color=colors)
plt.xlabel("Animal classes")
plt.ylabel("Number of samples")
plt.show()

X = df.drop(["animal_name", "class_type"], axis=1).values
y = df["class_type"].values

# Spliting to test and train data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Initializing the model
model = RandomForestClassifier()

# Training the model
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Printing the stats
print(f"Accuracy: {accuracy_score(y_pred, y_test)}")

plot_confusion_matrix(model, X_test, y_test, cmap=plt.cm.Blues)
plt.show()
