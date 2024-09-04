import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load the dataset
data = pd.read_csv('Crop_recommendation.csv')

# Split the dataset into input features (X) and target variable (y)
X = data[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
y = data['label']

# Create a Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the classifier
clf.fit(X, y)

# Save the trained model
pickle.dump(clf, open('model.pkl', 'wb'))
