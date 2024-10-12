import re
import string
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Load the dataset
df = pd.read_csv('SMSSpamCollection', sep='\t', names=['label', 'sms_message'])
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

X_train, X_test, y_train, y_test = train_test_split(df['sms_message'], df['label'], test_size=0.2, random_state=42)

vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

classifier = MultinomialNB()
classifier.fit(X_train_vectorized, y_train)

predictions = classifier.predict(X_test_vectorized)

accuracy = accuracy_score(y_test, predictions)
conf_matrix = confusion_matrix(y_test, predictions)
classification_rep = classification_report(y_test, predictions)

def accu():
    y=accuracy_score(y_test, predictions)
    return round(y,5)*100

print(f'Test Accuracy: {accuracy:.2f}')
print(f'Confusion Matrix: {conf_matrix}')
print(f'Classification Report: {classification_rep}')

def preprocess_text(text):
    text = text.lower()
    # Remove punctuation and numbers
    text = re.sub(f"[{re.escape(string.punctuation + '0123456789')}]", '', text)
    return text

def predict_spam_or_not_spam(new_sms):
    new_sms = preprocess_text(new_sms)
    
    new_sms_vectorized = vectorizer.transform([new_sms])
    
    prediction = classifier.predict(new_sms_vectorized)
    # Return the result
    return prediction[0], new_sms

def analyze_sentiment(text):
    sid = SentimentIntensityAnalyzer()
    sentiment_scores = sid.polarity_scores(text)
    
    if sentiment_scores['compound'] >= 0.05:
        return 'Positive'
    elif sentiment_scores['compound'] <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'
        

