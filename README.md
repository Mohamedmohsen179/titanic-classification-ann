# 🚢 Titanic Survival Prediction (ANN)

![Python](https://img.shields.io/badge/Python-3.x-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Status](https://img.shields.io/badge/Status-Completed-success)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📌 Overview

This project implements a **Machine Learning classification model** using an **Artificial Neural Network (ANN)** to predict whether a passenger survived the Titanic disaster.

It demonstrates a complete ML pipeline:

* Data preprocessing & cleaning
* Feature encoding & scaling
* Model building using Keras
* Training & evaluation
* Interactive prediction system

---

## 🧠 Model Architecture

```
Input Layer (7 features)
        ↓
Dense (16 neurons, ReLU)
        ↓
Dense (8 neurons, ReLU)
        ↓
Output (1 neuron, Sigmoid)
```

---

## 📊 Dataset Features

| Feature  | Description                   |
| -------- | ----------------------------- |
| Pclass   | Passenger class               |
| Sex      | Gender (0 = male, 1 = female) |
| Age      | Age of passenger              |
| SibSp    | Siblings/Spouses aboard       |
| Parch    | Parents/Children aboard       |
| Fare     | Ticket fare                   |
| Embarked | Port of embarkation           |

---

## ⚙️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* TensorFlow / Keras

---

## 🔄 Workflow

### 1. Data Loading

* Load dataset using pandas

### 2. Data Cleaning

* Drop irrelevant columns (Name, Ticket, Cabin)
* Handle missing values

### 3. Preprocessing

* Encode categorical variables
* Apply feature scaling (StandardScaler)

### 4. Model Training

* Build ANN with Dense layers
* Train using Adam optimizer

### 5. Evaluation

* Evaluate model performance on test data

### 6. Prediction

* Take user input
* Predict survival with confidence score

---

## 📈 Model Performance

```text
Accuracy: ~80% - 85%
```

---

## ▶️ How to Run

### 1. Install dependencies

```bash
pip install pandas numpy scikit-learn tensorflow matplotlib
```

### 2. Run the project

```bash
python main.py
```

### 3. Example Input

```text
3 0 22 1 0 7.25 0
```

---

## 🧪 Example Output

```text
Prediction: Survived
Confidence: 82.45%
```

---

## 📁 Project Structure

```
titanic-survival-ann/
│
├── main.py
├── titanic.csv
├── README.md
├── LICENSE
└── requirements.txt
```

---

## 🚀 Future Improvements

* Add confusion matrix visualization
* Hyperparameter tuning
* GUI interface (Tkinter / Streamlit)
* Model saving & loading

---

## 👨‍💻 Author

**Mohamed Mohsen**

---

## ⭐ Final Note

This project was developed as part of a **Machine Learning practical assignment**, demonstrating understanding of ANN-based classification systems.

---
