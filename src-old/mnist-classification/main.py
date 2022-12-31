from matplotlib import pyplot as plt
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

X, y = load_digits(return_X_y=True)

# Visualize the first digit
plt.figure(figsize=(20, 4))
for index, (image, label) in enumerate(zip(X[:5], y[:5])):
    plt.subplot(1, 5, index + 1)
    plt.imshow(image.reshape(8, 8), cmap="gray")
    plt.title(f"Number: {label}")

plt.show()

# Split into sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Create the model
model = RandomForestClassifier()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Get the scores
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
