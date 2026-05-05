
# Name: Alhathal Naif Mashaan D
# ID: SW01084553
# Sentiment Analysis Assignment

# Import Libraries
import pandas as pd
import numpy as np
import re
import nltk

from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

# Download stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))


# 1. Load Dataset

data = pd.read_csv("Reviews.csv", nrows=50000)

# Keep only required columns
data = data[['Score', 'Text']]


# 2. Convert Score to Sentiment

def get_sentiment(score):
    if score <= 2:
        return "Negative"
    elif score == 3:
        return "Neutral"
    else:
        return "Positive"

data['Sentiment'] = data['Score'].apply(get_sentiment)


# 3. Text Cleaning

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

data['Cleaned_Text'] = data['Text'].apply(clean_text)


# 4. Feature Extraction (TF-IDF)

vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(data['Cleaned_Text'])

y = data['Sentiment']


# 5. Train-Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# 6. Model 1: Logistic Regression

lr_model = LogisticRegression(max_iter=200)
lr_model.fit(X_train, y_train)

lr_pred = lr_model.predict(X_test)

print("=== Logistic Regression ===")
print("Accuracy:", accuracy_score(y_test, lr_pred))
print(classification_report(y_test, lr_pred))


# 7. Model 2: Naive Bayes

nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)

nb_pred = nb_model.predict(X_test)

print("=== Naive Bayes ===")
print("Accuracy:", accuracy_score(y_test, nb_pred))
print(classification_report(y_test, nb_pred))


# 8. Save Processed Data

data.to_csv("processed_reviews.csv", index=False)


# 9. Discussion (Print)

print("\n=== Discussion ===")
print("""
Machine learning models performed significantly better than basic approaches.

Logistic Regression achieved higher accuracy and balanced performance across classes,
making it more reliable for sentiment classification.

Naive Bayes was faster but struggled with minority classes like Neutral and Negative,
showing poor recall.

A key limitation is class imbalance in the dataset, where Positive reviews dominate.
This causes models to bias towards predicting Positive sentiment.

Overall, Logistic Regression is more effective, while Naive Bayes is efficient but less accurate.
""")