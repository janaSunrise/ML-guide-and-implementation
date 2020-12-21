import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
encoder = LabelEncoder()

column_names = [
    "class", "largest_spot_size", "spot_distribution", "activity",
    "evolution", "past24hour_flare_activity", "historically_complex",
    "region_became_historically_complex_on_pass", "area", "largest_spot_area",
    "common_flare", "moderate_flare", "severe_flare"
]

flare_1 = pd.read_csv("flare1.csv", sep=" ", names=column_names).apply(LabelEncoder().fit_transform)
flare_2 = pd.read_csv("flare2.csv", sep=" ", names=column_names).apply(LabelEncoder().fit_transform)

# Clean the datasets
df = pd.concat([flare_1, flare_2])

X = df.drop(["common_flare", "moderate_flare", "severe_flare"], axis=1)
y = df[["common_flare", "moderate_flare", "severe_flare"]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create Multiclassifier model
model = MultiOutputClassifier(SVC(kernel='linear'))

model.fit(X_train, y_train)

# Print stats
print(f"Accuracy: {model.score(X_test, y_test)}")
