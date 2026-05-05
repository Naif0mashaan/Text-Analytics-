# SENTIMENT ANALYSIS ASSIGNMENT
# Lexicon-Based Approach (TextBlob + VADER)
# Machine Learning Approach (Naive Bayes + SVM)

# Import required libraries
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from tabulate import tabulate
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC


# Sample dataset
data = [
("I love this product, it's amazing!", 'positive'),
("This product is terrible, I hate it.", 'negative'),
("It's okay, not bad but not great either.", 'neutral'),
("Best product ever, highly recommended!", 'positive'),
("I'm really disappointed with the quality.", 'negative'),
("So-so product, nothing special about it.", 'neutral'),
("The customer service was excellent!", 'positive'),
("I wasted my money on this useless product.", 'negative'),
("It's not the worst, but certainly not the best.", 'neutral'),
("I can't live without this product, it's a lifesaver!", 'positive'),
("The product arrived damaged and unusable.", 'negative'),
("It's average, neither good nor bad.", 'neutral'),
("Highly disappointed with the purchase.", 'negative'),
("The product exceeded my expectations.", 'positive'),
("It's just okay, nothing extraordinary.", 'neutral'),
("This product is excellent, it exceeded all my expectations!", 'positive'),
("I regret purchasing this product, it's a waste of money.", 'negative'),
("It's neither good nor bad, just average.", 'neutral'),
("Outstanding customer service, highly recommended!", 'positive'),
("I'm very disappointed with the quality of this item.", 'negative'),
("It's not the best product, but it gets the job done.", 'neutral'),
("This product is a game-changer, I can't imagine life without it!", 'positive'),
("I received a defective product, very dissatisfied.", 'negative'),
("It's neither great nor terrible, just okay.", 'neutral'),
("Fantastic product, I would buy it again in a heartbeat!", 'positive'),
("Avoid this product at all costs, complete waste of money.", 'negative'),
("It's decent, but nothing extraordinary.", 'neutral'),
("Impressive quality, exceeded my expectations!", 'positive'),
("I'm very unhappy with this purchase, total disappointment.", 'negative'),
("It's neither amazing nor terrible, somewhere in between.", 'neutral')
]


# --------------------------------------------------
# PART 1 : LEXICON BASED APPROACH
# --------------------------------------------------

print("\n==============================")
print("LEXICON BASED SENTIMENT ANALYSIS")
print("==============================\n")

table_data = [["Text","Actual Label","TextBlob Polarity","TextBlob Sentiment","VADER Compound","VADER Sentiment"]]

analyzer = SentimentIntensityAnalyzer()

for text, actual_label in data:

    # TextBlob analysis
    blob = TextBlob(text)
    tb_polarity = blob.sentiment.polarity

    if tb_polarity > 0:
        tb_label = "positive"
    elif tb_polarity < 0:
        tb_label = "negative"
    else:
        tb_label = "neutral"

    # VADER analysis
    vs = analyzer.polarity_scores(text)
    vader_compound = vs["compound"]

    if vader_compound > 0.05:
        vader_label = "positive"
    elif vader_compound < -0.05:
        vader_label = "negative"
    else:
        vader_label = "neutral"

    table_data.append([text,actual_label,tb_polarity,tb_label,vader_compound,vader_label])


# Display table
print(tabulate(table_data,headers="firstrow",tablefmt="grid"))


# Classification reports
actual_labels = [label for _,label in data]
tb_predictions = [row[3] for row in table_data[1:]]
vader_predictions = [row[5] for row in table_data[1:]]

print("\nClassification Report for TextBlob\n")
print(classification_report(actual_labels,tb_predictions))

print("\nClassification Report for VADER\n")
print(classification_report(actual_labels,vader_predictions))


# --------------------------------------------------
# PART 2 : MACHINE LEARNING APPROACH
# --------------------------------------------------

print("\n==============================")
print("MACHINE LEARNING SENTIMENT ANALYSIS")
print("==============================\n")

texts = [text for text,_ in data]
labels = [label for _,label in data]


# Split dataset
X_train,X_test,y_train,y_test = train_test_split(
texts,labels,test_size=0.4,random_state=42
)


# Feature extraction
vectorizer = CountVectorizer()

X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)


# Initialize models
nb_classifier = MultinomialNB()
svm_classifier = SVC(kernel="linear")


# Train models
nb_classifier.fit(X_train,y_train)
svm_classifier.fit(X_train,y_train)


# Predictions
nb_predictions = nb_classifier.predict(X_test)
svm_predictions = svm_classifier.predict(X_test)


# Classification reports
print("Classification Report for Naive Bayes\n")
print(classification_report(y_test,nb_predictions))

print("Classification Report for SVM\n")
print(classification_report(y_test,svm_predictions))