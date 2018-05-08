from load_model import predict
from flask import Flask, render_template, request



app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def main():
    if request.method == "POST":
        cement = request.form['cement']
        return cement
        # fine = request.form['fine']
        # coarse = request.form['coarse']
        # age = request.form['age']
        # strength = request.form['strength']
        # result = predict(cement, fine, coarse, age, strength, LINREG=True)
        # return render_template('index.html', quatity=result)
    return render_template('index.html')
