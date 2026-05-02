# 🚢 Titanic Survival Prediction using ANN

## 📌 Overview

This project builds a **classification model** using an **Artificial Neural Network (ANN)** to predict whether a passenger survived the Titanic disaster.

The workflow includes:

* Data loading and exploration
* Data cleaning and preprocessing
* Feature encoding and scaling
* Model building using TensorFlow/Keras
* Training and evaluation
* Making predictions on new data

---

## 🧠 Model Architecture

The ANN model consists of:

* Input layer (based on dataset features)
* Hidden Layer 1: 16 neurons (ReLU)
* Hidden Layer 2: 8 neurons (ReLU)
* Output Layer: 1 neuron (Sigmoid)

---

## ⚙️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* TensorFlow / Keras

---

## 📊 Dataset

The dataset used is the **Titanic dataset**, which includes features such as:

* Passenger Class (Pclass)
* Sex
* Age
* SibSp (siblings/spouses aboard)
* Parch (parents/children aboard)
* Fare
* Embarked

---

## 🔄 Workflow

1. **Load Data**

   * Read dataset using pandas

2. **Data Cleaning**

   * Drop unnecessary columns (Name, Ticket, Cabin)
   * Handle missing values

3. **Preprocessing**

   * Encode categorical variables
   * Scale features using StandardScaler

4. **Train-Test Split**

   * 80% training, 20% testing

5. **Model Training**

   * Train ANN using Adam optimizer

6. **Evaluation**

   * Measure accuracy on test data

7. **Prediction**

   * Accept user input and predict survival

---

## 📈 Model Performance

The model achieves approximately:

```text id="acc1"
~80% - 85% accuracy
```

---

## ▶️ How to Run

1. Install dependencies:

```bash id="run1"
pip install pandas numpy scikit-learn tensorflow matplotlib
```

2. Run the script:

```bash id="run2"
python main.py
```

3. Enter input values when prompted:

```text id="run3"
Pclass Sex Age SibSp Parch Fare Embarked
```

Example:

```text id="run4"
3 0 22 1 0 7.25 0
```

---

## 📌 Example Output

```text id="out1"
Prediction: Survived
Confidence: 82.45%
```

---

## 🧪 Future Improvements

* Add confusion matrix visualization
* Improve feature engineering
* Add GUI interface
* Hyperparameter tuning

---

## 👨‍💻 Author

Mohamed Mohsen

---

## ⭐ Notes

This project is part of a Machine Learning practical assignment focusing on ANN classification.
