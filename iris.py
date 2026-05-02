import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 1. Load dataset
data = load_iris()
X = data.data
y = data.target

# 2. Preprocessing (Scaling)
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 3. Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

# 4. Build ANN
model = Sequential()

model.add(Dense(16, activation='relu', input_shape=(4,)))
model.add(Dense(8, activation='relu'))
model.add(Dense(3, activation='softmax'))

# 5. Compile
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 6. Training
history = model.fit(X_train, y_train, epochs=50, batch_size=8)

# 7. Evaluate
loss, acc = model.evaluate(X_test, y_test)
print("Accuracy =", acc)

# 8. Plot accuracy
plt.plot(history.history['accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show()

# 9. Prediction 

classes = ["Setosa", "Versicolor", "Virginica"]

try:
    sample = input("Enter 4 values separated by space: ")

    values = [float(x) for x in sample.split()]

    if len(values) != 4:
        print("Please enter exactly 4 values")
    else:
        for v in values:
            if v < 0 or v > 10:
                raise ValueError("Values out of range")

        sample = np.array(values).reshape(1, -1)

        sample = scaler.transform(sample)

        prediction = model.predict(sample)

        predicted_class = np.argmax(prediction)

        print("Predicted class:", classes[predicted_class])

except:
    print("Please enter valid numeric values")