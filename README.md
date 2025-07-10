# ✈️ US Airways Customer Tweets Sentiment Analysis

## 📌 Problem Statement

Airlines receive thousands of customer opinions via social media platforms like Twitter. Manually sorting and analyzing these tweets is labor-intensive and inefficient. The objective of this project is to build a sentiment analysis model that classifies customer reviews into Positive, Negative, or Neutral sentiments using natural language processing (NLP) techniques.

---

## 📊 Dataset

- **File**: `Dataset - US_Airways.csv`
- **Columns**:
  - `tweet_id`
  - `airline_sentiment` (target label)
  - `text` (actual tweet content)
  - `airline`
  - `tweet_created`
  - `negativereason` (if applicable)

---

## 🛠️ Tools & Technologies

- Python 3.x
- pandas, NumPy
- scikit-learn
- nltk, seaborn, matplotlib
- WordCloud, Logistic Regression, Random Forest, Naive Bayes

---

## 🔁 Workflow

```python
# 1. Import Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import re
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

# 2. Load Dataset
df = pd.read_csv('Dataset - US_Airways.csv')

# 3. Preprocessing
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    words = [w for w in text.split() if w not in stop_words]
    return " ".join(words)

df['clean_text'] = df['text'].apply(clean_text)

# 4. Vectorization
vectorizer = CountVectorizer(max_features=3000)
X = vectorizer.fit_transform(df['clean_text']).toarray()
y = df['airline_sentiment']

# 5. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Model Training & Evaluation

# Logistic Regression
lr = LogisticRegression()
lr.fit(X_train, y_train)
pred_lr = lr.predict(X_test)
print("Logistic Regression:\n", classification_report(y_test, pred_lr))

# Naive Bayes
nb = MultinomialNB()
nb.fit(X_train, y_train)
pred_nb = nb.predict(X_test)
print("Naive Bayes:\n", classification_report(y_test, pred_nb))

# Random Forest
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
pred_rf = rf.predict(X_test)
print("Random Forest:\n", classification_report(y_test, pred_rf))
```

---

## ✅ Results

- **Logistic Regression** showed the most consistent performance across sentiment categories.
- **Naive Bayes** provided a good baseline and performed well on Negative sentiment detection.
- **Random Forest** offered high accuracy but required tuning for imbalanced classes.

---

## 📈 Visualizations

- Sentiment distribution bar chart
- Word cloud of most common negative/positive words
- Correlation heatmap of feature vectors

---

## 📁 Repository Structure

```
├── Session Notebook - Airline Customer Review Sentiment Analysis.ipynb
├── Dataset - US_Airways.csv
├── README.md
```

---

## 👨‍💻 Author

**Suhaib Khalid**  
AI/ML Practitioner 
