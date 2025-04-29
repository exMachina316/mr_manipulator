import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the dataset
data_dict = pickle.load(open('data.pickle', 'rb'))
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Split the dataset
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.4, shuffle=True, stratify=labels)

# Train the classifier
model = RandomForestClassifier()
model.fit(x_train, y_train)

y_predict = model.predict(x_test)
score = accuracy_score(y_predict, y_test)

print(f'{score * 100:.2f}% of samples were classified correctly!')

# Save the trained model
with open('model.2.0.2.p', 'wb') as f:
    pickle.dump({'model': model}, f)

print("Model trained and saved!")
