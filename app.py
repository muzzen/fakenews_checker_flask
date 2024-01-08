from flask import Flask, request, render_template
import pickle

# initiating the flask
app = Flask(__name__)

# load the classifier
classifier = pickle.load(
    open('clf.pkl', 'rb')
)

vectorizer = pickle.load(
    open('vec.pkl','rb')
)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        input_text = request.form['news_text'].lower()
              
        word_list = set(input_text.split())
        word = len(word_list)

        input_text_vec = vectorizer.transform([input_text])
        
        token = len([w for w in word_list if w in vectorizer.vocabulary_])

        input_text = classifier.predict(input_text_vec)

    
        return render_template('home.html', input_text = input_text[0],
                               word = word, token = token)
    
    else:
        return render_template('home.html')
    


if __name__ == '__main__':
    app.run(port=8000)