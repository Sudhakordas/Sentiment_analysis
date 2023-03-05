from flask import Flask, render_template, request
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.linear_model import PassiveAggressiveClassifier
import pickle
import pandas as pd
import re

app = Flask(__name__)

# making cleaning function
def find_the (text):
    while len(re.findall('##' , text)) != 0:
        text = re.sub('##' , '#' ,text)
    while len(re.findall('@@' , text)) != 0:
        text = re.sub('@@' , '@' ,text)

    no_h = len(re.findall('#',text))
    #make sure the @ is changed for at
    no_a = len(re.findall('@',text))
    
    text = re.sub('@(\s|\.|\?|,|;|:|!|\(|\))', 'at' , text)
    text = re.sub('\s' , ' ' , text)
    text = re.sub('\S@' , ' @' , text)
    text = re.sub('\S#' , ' #' , text)

    #while len(re.findall('\s\s' , text)) != 0:
    #    text = re.sub('\s\s' , '\s' ,text)

    #none_repeat = text
    text_a = text.split()         
    return text

def removing_url(text):
#replace URL of a text
    cltext = re.sub('http[^\s]+','',text)
    return cltext

def text_processing(text):
    review = re.sub('[^a-zA-Z]', ' ', text)
    review = review.lower()
    review = review.split()    
    review = [ps.stem(word) for word in review if not word in stop_words]
    review = ' '.join(review)
    return review

def text_cleaning(text):
  text = find_the(text)
  text = removing_url(text)
  text = text_processing(text)

  return text


# load the model from disk
model = pickle.load(open('svc_model.pkl', 'rb'))
vector = pickle.load(open('vectorizer.pkl', 'rb'))
stop_words = pickle.load(open('stopwords.pkl', 'rb'))
ps = pickle.load(open('stem.pkl', 'rb'))

class_labels = {0:'Negative',
                1: 'Positive'}

def sentiment_analysis(message):
    input_data = [message]
    vectorized_input_data = vector.transform(input_data)
    prediction = model.predict(vectorized_input_data)
    prediction = class_labels[prediction[0]]
    return prediction

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        pred = sentiment_analysis(message)
        print(pred)
        return render_template('index.html', prediction=pred)
    else:
        return render_template('index.html', prediction="Something went wrong")

if __name__ == '__main__':
    app.run(debug=True)