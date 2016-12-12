from flask import render_template, redirect, request
from app import app, models, showcase
#from .forms import CustomerForm
#from models import *
# Access the models file to use SQL functions


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['GET', 'POST'])
def analyze():
    tweet = request.form['tweet']
    result = showcase.RUNME(tweet)
    print(result)
    #send_value = "FAVORS " + result
    return render_template('index.html', result=result, tweet=tweet)

# @app.route('/customers')
# def display_customer():
#     #Retreive data from database to display
#     customers = retrieve_customers()
#     return render_template('home.html',
#                             customers=customers)
