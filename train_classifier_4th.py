import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Step 1: Load raw data
with open('./data.pickle', 'rb') as f:
    data_dict = pickle.load(f)

raw_data = data_dict['data']
raw_labels = data_dict['labels']

print("Total raw samples:", len(raw_data))

# Step 2: Filter valid samples only
filtered_data = []
filtered_labels = []

for d, l in zip(raw_data, raw_labels):
    if isinstance(d, list) and len(d) == 42 and all(isinstance(v, (float, int)) for v in d):
        filtered_data.append(d)
        filtered_labels.append(l)

print("Total valid samples:", len(filtered_data))

# Error handling if no valid samples
if len(filtered_data) == 0:
    raise ValueError("No valid samples found. Please check your dataset format.")

# Step 3: Convert to numpy arrays
data = np.array(filtered_data, dtype=np.float32)
labels = np.array(filtered_labels)

# Step 4: Train-test split
x_train, x_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, shuffle=True, stratify=labels
)

# Step 5: Train the model
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Step 6: Evaluate
y_predict = model.predict(x_test)
score = accuracy_score(y_predict, y_test)
print(f"{score * 100:.2f}% of samples were classified correctly!")

# Step 7: Save the model
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)
