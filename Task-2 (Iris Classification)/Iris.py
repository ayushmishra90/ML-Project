import joblib
from sklearn import datasets, svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os

# Get the directory of the current script
script_directory = os.path.dirname(os.path.abspath(__file__))

# Define paths for model and accuracy files
model_path = os.path.join(script_directory, 'svm_model.pkl')
accuracy_path = os.path.join(script_directory, 'accuracy.txt')

# Load and train the model
iris = datasets.load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = svm.SVC()
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, model_path)

# Save the accuracy
accuracy = accuracy_score(y_test, model.predict(X_test))
with open(accuracy_path, 'w') as f:
    f.write(f'Accuracy: {accuracy:.2f}')

print(f'Model saved at: {model_path}')
print(f'Accuracy saved at: {accuracy_path}')
print(f'Model saved with accuracy: {accuracy:.2f}')
