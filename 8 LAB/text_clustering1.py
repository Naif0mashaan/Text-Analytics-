import numpy as np
import pandas as pd
import re
import nltk

from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from collections import Counter

nltk.download('stopwords')
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)
dataset = [
    "I love playing football on the weekends",
    "I enjoy hiking and camping in the mountains",
    "I like to read books and watch movies",
    "I prefer playing video games over sports",
    "I love listening to music and going to concerts"
]
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(dataset)

km = KMeans(n_clusters=2, random_state=42)
y_pred = km.fit_predict(X)

print("TF-IDF WITHOUT PREPROCESSING:")
for doc, cluster in zip(dataset, y_pred):
    print(cluster, "->", doc)

# Purity
purity = max(Counter(y_pred).values()) / len(y_pred)
print("Purity:", purity)
processed_dataset = [preprocess(doc) for doc in dataset]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(processed_dataset)

km = KMeans(n_clusters=2, random_state=42)
y_pred = km.fit_predict(X)

print("\nTF-IDF WITH PREPROCESSING:")
for doc, cluster in zip(dataset, y_pred):
    print(cluster, "->", doc)

purity = max(Counter(y_pred).values()) / len(y_pred)
print("Purity:", purity)
tokenized = [preprocess(doc).split() for doc in dataset]

model = Word2Vec(sentences=tokenized, vector_size=100, window=5, min_count=1)

# Document vectors (average of word vectors)
X = np.array([
    np.mean([model.wv[word] for word in words], axis=0)
    for words in tokenized
])

km = KMeans(n_clusters=2, random_state=42)
y_pred = km.fit_predict(X)

print("\nWORD2VEC WITH PREPROCESSING:")
for doc, cluster in zip(dataset, y_pred):
    print(cluster, "->", doc)

purity = max(Counter(y_pred).values()) / len(y_pred)
print("Purity:", purity)
df = pd.read_csv("customer_complaints_1.csv")
print(df.head())  # Check column names

# Use correct column name (likely 'text', lowercase)
texts = df['text'].astype(str).tolist()

# Preprocess
processed_texts = [preprocess(t) for t in texts]

# TF-IDF clustering
vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(processed_texts)

km = KMeans(n_clusters=3, random_state=42)
y_pred = km.fit_predict(X)

# Show first 10 results
print("\nCustomer Complaints Clustering:")
for i in range(10):
    print(y_pred[i], "->", texts[i])

# Top words per cluster
order_centroids = km.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names_out()

print("\nTop words per cluster:")
for i in range(3):
    print("\nCluster", i)
    for ind in order_centroids[i, :10]:
        print(terms[ind])