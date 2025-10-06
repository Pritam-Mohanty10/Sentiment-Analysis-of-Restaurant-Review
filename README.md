# 🍽️ Sentiment Analysis of Restaurant Reviews  
Automatically decide whether a restaurant review is **positive** or **negative** using a lightweight, fully-self-contained NLP pipeline.

---

## 🧠 Why This Project
Online reviews shape customer choices, but reading them all is impossible.  
This repo gives you a clean, reproducible baseline that turns free-form text into actionable sentiment in **&lt;5 min**—no GPU, no cloud, no prior NLP knowledge required.

---

## 🧪 Methodology
1. **Text Cleaning**  
   - Lower-casing, regex punctuation strip, Porter stemming  
   - NLTK stop-word list + custom domain stop-words  
2. **Feature Engineering**  
   - Bag-of-Words (`CountVectorizer`, max-features = 1 500, n-gram range = (1,2))  
3. **Model**  
   - Logistic Regression (C = 1.0, liblinear solver)  
4. **Evaluation**  
   - 70/30 train-test split (stratified)  
   - Metrics: Accuracy, Precision, Recall, F1, Confusion Matrix  
   - 10-fold cross-validation for stability check  

---

## 📊 Dataset Snapshot
| Source | Size | Balance | Avg. Tokens | Vocab |
|--------|------|---------|-------------|-------|
| Kaggle “Restaurant_Reviews.tsv” | 1 000 | 50 % positive / 50 % negative | 28 | 2 865 |

Example rows  
| Review | Label |
|--------|-------|
| “Wow... Loved this place.” | 1 |
| “Crust is not good.” | 0 |

---

## 🚀 Quick Start (≤2 min)
```bash
# 1. Clone
git clone https://github.com/Pritam-Mohanty10/Sentiment-Analysis-of-Restaurant-Review.git
cd Sentiment-Analysis-of-Restaurant-Review

# 2. Create venv (optional but recommended)
python -m venv venv && source venv/bin/activate   # Win: venv\Scripts\activate

# 3. Install
pip install -r requirements.txt
| Metric          | Score          |
| --------------- | -------------- |
| Accuracy        | 85.3 %         |
| Precision (pos) | 0.86           |
| Recall (pos)    | 0.84           |
| F1 (pos)        | 0.85           |
| Cross-val mean  | 84.7 % ± 2.1 % |
Sentiment-Analysis-of-Restaurant-Review/
├── restaurant_reviews.tsv      # raw data
├── main.py                     # train + evaluate + save png
├── predict.py                  # single-review CLI helper
├── requirements.txt            # ≤10 lightweight deps
├── confusion_matrix.png        # auto-generated after run
└── README.md                   # this file
# predict.py
from joblib import load
vec = load('vectorizer.joblib')
clf = load('classifier.joblib')

review = "The pasta was heavenly and the staff remembered my birthday!"
X = vec.transform([review])
print("positive" if clf.predict(X)[0] else "negative")
# → positive
python>=3.7
nltk
scikit-learn
pandas
numpy
matplotlib
seaborn
joblib

# 4. Train + Evaluate
python main.py
# → console prints 85 % accuracy and saves confusion_matrix.png
