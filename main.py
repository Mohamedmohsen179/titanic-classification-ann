import pandas as pd
import numpy as np

# ================================
# 1. Load and explore data
# ================================
df = pd.read_csv("titanic.csv")

print(df.head())
print("="*50)
print(df.columns)
print("="*50)
print(df.info())
print("="*50)
print(df.isnull().sum())

# ================================
# 2. Cleaning & Preprocessing
# ================================

# Drop unnecessary columns
df = df.drop(['Name', 'Ticket', 'Cabin'], axis=1)

# Convert categorical to numeric
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

# Fill missing values
df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# Encode Embarked
df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

# Split features and label
X = df.drop('Survived', axis=1)
y = df['Survived']

# ================================
# 3. Scaling
# ================================
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X = scaler.fit_transform(X)

# ================================
# 4. Train-Test Split
# ================================
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ================================
# 5. Build ANN Model
# ================================
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()

model.add(Dense(16, activation='relu', input_shape=(X.shape[1],)))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# ================================
# 6. Training & Evaluation
# ================================
history = model.fit(X_train, y_train, epochs=50, batch_size=8)

loss, acc = model.evaluate(X_test, y_test)
print(f"\nModel Accuracy: {acc * 100:.2f}%")

# ================================
# 7. Prediction (User Input Loop)
# ================================
classes = ["Not Survived", "Survived"]

print("\n" + "="*40)
print("Titanic Survival Prediction")
print("="*40)

while True:
    try:
        sample = input("\nEnter values (Pclass Sex Age SibSp Parch Fare Embarked) or 'q' to exit: ")

        # خروج
        if sample.lower() == 'q':
            print("\nExiting...\n")
            break

        values = [float(x) for x in sample.split()]

        # Check correct number of inputs
        if len(values) != X.shape[1]:
            print(f"\nPlease enter exactly {X.shape[1]} values\n")
            continue

        # Convert to array
        sample_array = np.array(values).reshape(1, -1)

        # Apply scaling
        sample_scaled = scaler.transform(sample_array)

        # Predict
        prediction = model.predict(sample_scaled)
        prob = prediction[0][0]

        result = 1 if prob > 0.5 else 0
        confidence = prob if result == 1 else (1 - prob)

        # Display input
        print("\nInput Data:")
        print(f"Pclass: {values[0]}")
        print(f"Sex: {'Male' if values[1] == 0 else 'Female'}")
        print(f"Age: {values[2]}")
        print(f"Fare: {values[5]}")

        # Display result
        print("\nPrediction:")
        print(classes[result])
        print(f"Confidence: {confidence * 100:.2f}%")

        if result == 1:
            print("\nLikely to survive.")
        else:
            print("\nUnlikely to survive.")

        print("\n" + "="*40)

    except:
        print("\nPlease enter valid numeric values\n")