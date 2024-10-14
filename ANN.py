import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# Load feature data from Excel
feature_data = pd.read_excel('training_data/feature_data.xlsx')  # Update with your feature data file path

# Load training data from Excel, including three columns for win, loss, knockout
training_data = pd.read_excel('training_data/training_data.xlsx')  # Update with your training data file path


# Assuming that the 'outcome' column in the training data contains the 3-digit outcome vector
outcome_data = training_data['outcome']


# Split the data into training, cross-validation, and testing sets
X_train, X_temp, y_train, y_temp = train_test_split(
    feature_data, outcome_data, test_size=0.4, random_state=42)

X_cv, X_test, y_cv, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42)

# Preprocess the data
# Perform one-hot encoding for the outcome data
encoder = OneHotEncoder(sparse=False, categories='auto')  # You can use 'auto' to automatically detect categories
y_train_encoded = encoder.fit_transform(y_train.values.reshape(-1,1))
y_cv_encoded = encoder.transform(y_cv.values.reshape(-1,1))
y_test_encoded = encoder.transform(y_test.values.reshape(-1,1))

# Normalise the features
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_cv = scaler.transform(X_cv)
X_test = scaler.transform(X_test)

# Define the model
model = keras.Sequential()
model.add(keras.layers.Input(shape=(44,)))
model.add(keras.layers.Dense(units=64, activation='relu'))
model.add(keras.layers.Dense(units=32, activation='relu'))
model.add(keras.layers.Dense(units=16, activation='relu'))
model.add(keras.layers.Dense(units=3, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train_encoded, epochs=10, batch_size=32, validation_data=(X_cv, y_cv_encoded))

# Plot training and validation loss
plt.figure(figsize=(8, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Evaluate the model on the testing data
loss, accuracy = model.evaluate(X_test, y_test_encoded)

# Predict on the test data
y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test_encoded, axis=1)

# Calculate precision, recall, and F1-score
report = classification_report(y_true, y_pred, target_names=["Win", "Knockout", "Loss"])

print(f'Accuracy on test data: {accuracy * 100:.2f}%')
print(report)
