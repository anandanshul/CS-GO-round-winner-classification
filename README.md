

# 🎯 CS:GO Round Winner Prediction

## 📌 Project Overview
This project predicts the **round winner (Counter-Terrorists vs Terrorists)** in **Counter-Strike: Global Offensive (CS:GO)** using supervised machine learning models.  

We leverage the **CS:GO Round Snapshots dataset** from Kaggle and apply multiple ML classifiers (`Logistic Regression`, `Decision Tree`, `Random Forest`, `XGBoost`, and `MLP Neural Network`).  
The workflow includes **data cleaning, encoding, scaling, model training, and evaluation**.  

---

## 🕹️ About CS:GO
Counter-Strike: Global Offensive (CS:GO) is a tactical FPS where two teams compete:  
- **Terrorists (T)** must plant and detonate a bomb.  
- **Counter-Terrorists (CT)** must stop them or defuse the bomb.  

The outcome of each round depends on the **in-game economy, weapons, and decisions**, making it ideal for predictive modeling.

---

## 📂 Dataset
- **Source:** [CSGO Round Snapshots (Kaggle)](https://www.kaggle.com/)  
- **File used:** `csgo_round_snapshots.csv`  
- **Features:**  
  - Map information  
  - Bomb plant status  
  - Player/Team equipment values  
  - Economy-related stats  
  - Round outcomes (target variable: `round_winner`)  

---

## ⚙️ Data Preprocessing
1. **Drop redundant columns** with only one unique value.  
2. **Handle missing values** (checked with `.isnull().sum()`).  
3. **Label Encoding:**  
   - `map`  
   - `bomb_planted`  
   - `round_winner`  
4. **Feature Scaling:**  
   - Applied `StandardScaler` to normalize numeric values.  
5. **Train-Test Split:**  
   - `train_test_split(test_size=0.2, random_state=42)`  

---

## 🤖 Models Implemented
The following classifiers were trained and evaluated:
- **Logistic Regression**  
- **Decision Tree Classifier**  
- **Random Forest Classifier**  
- **XGBoost Classifier**  
- **MLP Neural Network (Multi-Layer Perceptron)**  

### Example: Random Forest Training
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

model = RandomForestClassifier()
model.fit(x_train, y_train)
pred = model.predict(x_test)

print("Classification Report:\n", classification_report(y_test, pred))
print("Accuracy:", accuracy_score(y_test, pred))
````

---

## 📊 Results

* All models were evaluated using **Accuracy Score** and **Classification Report (Precision, Recall, F1-score)**.
* Among the models, **Random Forest Classifier** achieved the **best performance**.

**Best Accuracy:**

```text
Optimal accuracy for Random Forest Classifier: ~XX%  (replace with your output)
```

---

## 🚀 How to Run

1. Clone this repository:

   ```bash
   git clone https://github.com/yourusername/csgo-round-winner-prediction.git
   cd csgo-round-winner-prediction
   ```
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
3. Run the notebook:

   ```bash
   jupyter notebook csgo-round-winner-prediction.ipynb
   ```

---

## 🛠️ Tech Stack

* **Languages:** Python 3
* **Libraries:**

  * `numpy`, `pandas` → Data processing
  * `scikit-learn` → Preprocessing, ML models, evaluation
  * `xgboost` → Gradient boosting classifier
  * `matplotlib`, `seaborn` → Visualization

---

## 🔮 Future Improvements

* Perform **hyperparameter tuning** for better model optimization.
* Apply **cross-validation** to validate model generalizability.
* Use **deep learning (LSTM/GRU)** to model temporal round sequences.
* Deploy as an **API** with Flask/FastAPI for real-time predictions.

