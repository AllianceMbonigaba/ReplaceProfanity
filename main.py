from flask import Flask,render_template,request
from datamining_systemimplementation import remove_badwords, get_sentiment_score, get_only_profanity

app = Flask(__name__)

@app.route("/")
def index():
    return render_template('index.html') 

@app.route("/predicts/", methods=['POST', 'GET'])
def predicts():
    if request.method == 'POST':
        text = request.form['subject']
        profanity = get_only_profanity(text)
        sentiment = get_sentiment_score(text)
        total_profanity_words = len(profanity)

        # detect from the model
        result = remove_badwords(text)
        
        return render_template('predicts.html', text = result, profanity = profanity, sentiment= sentiment, total_p = total_profanity_words)
    
    if request.method == 'GET':

        return render_template('index.html') 




# test the app
if __name__ == '__main__':
    app.run()