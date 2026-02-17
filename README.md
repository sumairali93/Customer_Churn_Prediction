# 🏦 Customer Churn Prediction using ANN

A deep learning project that predicts whether a bank customer will churn (leave the bank) using an Artificial Neural Network built with TensorFlow/Keras. Trained on 10,000 customer records and achieved **86.15% accuracy** on the test set.

---

## 📌 Problem Statement

Customer churn is a critical business problem in banking. Losing customers is costly — acquiring new ones is 5–7x more expensive than retaining existing ones. This project builds a binary classifier to identify at-risk customers so the bank can take proactive retention actions.

---

## 📊 Dataset

- **Source:** [Kaggle — Credit Card Customer Churn Prediction](https://www.kaggle.com/datasets/rjmanoj/credit-card-customer-churn-prediction)
- **File:** `Churn_Modelling.csv`
- **Size:** 10,000 records × 14 features
- **Target:** `Exited` (1 = churned, 0 = retained)

### Features Used

| Feature | Description |
|---------|-------------|
| CreditScore | Customer's credit score |
| Geography | Country (France, Germany, Spain) |
| Gender | Male / Female |
| Age | Customer's age |
| Tenure | Years with the bank |
| Balance | Account balance |
| NumOfProducts | Number of bank products used |
| HasCrCard | Has credit card (1/0) |
| IsActiveMember | Active member (1/0) |
| EstimatedSalary | Estimated annual salary |

> `RowNumber`, `CustomerId`, and `Surname` were dropped as they carry no predictive value.

---

## 🔍 Data Analysis

```
Dataset shape: (10000, 14)
No duplicate rows found
Class distribution:
  Not churned (0): 7963
  Churned (1):     2037
```

**Geography distribution:**
- France: 5014
- Germany: 2509
- Spain: 2477

**Gender distribution:**
- Male: 5457
- Female: 4543

---

## ⚙️ Preprocessing Pipeline

1. **Dropped** non-predictive columns: `RowNumber`, `CustomerId`, `Surname`
2. **One-Hot Encoding** applied to `Geography` and `Gender` (`drop_first=True` to avoid multicollinearity)
3. **Train/Test Split:** 80% train, 20% test (`random_state=1`)
4. **Feature Scaling:** MinMaxScaler fitted on training data only, applied to both train and test sets

---

## 🧠 Model Architecture

```
Model: Sequential ANN
_____________________________________________
 Layer          Output Shape    Parameters
=============================================
 Dense (ReLU)   (None, 11)      132
 Dense (ReLU)   (None, 11)      132
 Dense (Sigmoid)(None, 1)       12
=============================================
 Total params: 276
```

- **Input:** 11 features (after encoding)
- **Hidden layers:** 2 × Dense(11, activation='relu')
- **Output:** Dense(1, activation='sigmoid') — binary classification
- **Optimizer:** Adam
- **Loss:** Binary Crossentropy
- **Metric:** Accuracy
- **Callback:** EarlyStopping (monitor=val_loss, patience=5, restore_best_weights=True)

---

## 📈 Results

| Metric | Value |
|--------|-------|
| Test Accuracy | **86.15%** |
| Training Epochs | 55 (stopped early) |

The training vs validation accuracy and loss curves confirmed the model generalizes well without significant overfitting.

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python | Core language |
| Pandas | Data loading & manipulation |
| NumPy | Array operations & thresholding |
| Scikit-learn | Train/test split, MinMaxScaler, accuracy_score |
| TensorFlow / Keras | ANN model (Sequential, Dense, EarlyStopping) |
| Matplotlib | Plotting accuracy & loss curves |
| Kaggle API | Dataset download |

---

## 📁 Project Structure

```
Customer_Churn_Prediction/
│
├── customer_churn_prediction.ipynb   # Full notebook: EDA → preprocessing → model → evaluation
├── requirements.txt                   # Python dependencies
└── README.md                          # Project documentation
```

---

## 🚀 How to Run

**1. Clone the repository**
```bash
git clone https://github.com/sumairali93/Customer_Churn_Prediction.git
cd Customer_Churn_Prediction
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Set up Kaggle API**

Place your `kaggle.json` API key in the project root, then run the notebook — it will auto-download the dataset.

> Get your API key from: https://www.kaggle.com/settings → API → Create New Token

**4. Run the notebook**
```bash
jupyter notebook customer_churn_prediction.ipynb
```

---

## 💡 Key Learnings

- Categorical encoding using `pd.get_dummies()` with `drop_first=True` prevents the dummy variable trap
- Feature scaling with MinMaxScaler is critical for neural network convergence
- EarlyStopping effectively prevents overfitting — model stopped at epoch 55 out of max 100
- Class imbalance (80% retained vs 20% churned) is an important consideration for real-world deployment

---

## 👤 Author

**Sumair Ali**
- GitHub: [@sumairali93](https://github.com/sumairali93)
- LinkedIn: [linkedin.com/in/sumairali93](https://linkedin.com/in/sumairali93)
