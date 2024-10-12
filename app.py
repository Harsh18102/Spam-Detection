# app.py

from flask import Flask, render_template, request
from moye import preprocess_text, predict_spam_or_not_spam, analyze_sentiment,accu
import re

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the SMS from the form
        new_sms = request.form['sms']
        new_sms = preprocess_text(new_sms)
        result, preprocessed_sms = predict_spam_or_not_spam(new_sms)
        sentiment = analyze_sentiment(new_sms)
        acc=accu()

        response = {
            'classification': "Spam" if result == 1 else "Not Spam",
            'sentiment': sentiment,
            'accuracy':acc
        }

        return render_template('result.html', result=response)

if __name__ == '__main__':
    app.run(debug=True)
