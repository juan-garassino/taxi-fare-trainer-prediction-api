
from flask import Flask, escape, request

import pandas as pd

import joblib

# create flask app
app = Flask(__name__)


@app.route('/')
def hello():
    # get param from http://127.0.0.1:5000/?name=value
    name = request.args.get("name", "World")
    return f'Hello, {escape(name)}!'


"""
MVC

controllers/
constrolles/user_controller.py => create a new user / login
constrolles/tweet_controller.py => create a tweet / list all the tweets

models/
models/user.py User => describe the attributes of the user (id, login + creation_date + picture)
models/tweet.py Tweet => describe the attributes of the tweet (id, user_id, content + sent_date + number_retweets)

views/
views/view_all_tweets.py =>


# USER CONTROLLER

@app.route('/create_login')
def create_user():
    pass  # goal : tells the model to create a user

@app.route('/login')
def check_login():
    pass  # controller asks the model to verify the account
"""


@app.route('/toto')
def hello_toto():
    return '''
    <!DOCTYPE>
    <html>
        <head>
            <title>My super page</title>
        </head>
        <body>
        Anything
            <div>
                This is a Le Wagon API site, please use rather the /predict_fare entry point
                <img src="https://dwj199mwkel52.cloudfront.net/assets/core/home/coding-school-that-cares-alumni-025e665def0e2f5a9a539cd2f8762fedbd4c5074a725ebed08570a5bdacc45f7.jpg">
            </div>
            <div>
                <!-- List of all the tweets -->
                <div>
                    <div class="tweet-title">
                        <div class="tweet-content">abc</div>
                    </div>
                    <div class="tweet-title">
                        <div class="tweet-content">abc</div>
                    </div>
                    <div class="tweet-title">
                        <div class="tweet-content">abc</div>
                    </div>
                    <div class="tweet-title">
                        <div class="tweet-content">abc</div>
                    </div>
                </div>
            </div>
        </body>
    </html>
    '''


@app.route('/predict_fare', methods=['GET'])
def predict_fare():

    # get request arguments
    key = request.args.get('key')
    pickup_datetime = request.args.get('pickup_datetime')
    pickup_longitude = float(request.args.get('pickup_longitude'))
    pickup_latitude = float(request.args.get('pickup_latitude'))
    dropoff_longitude = float(request.args.get('dropoff_longitude'))
    dropoff_latitude = float(request.args.get('dropoff_latitude'))
    passenger_count = int(request.args.get('passenger_count'))

    # build X ⚠️ beware to the order of the parameters ⚠️
    X = pd.DataFrame({
        "Unnamed: 0": [1],  # These are not used by the model
        "key": [key],  # but they are required by the pipeline as it is coded
        "pickup_datetime": [pickup_datetime],
        "pickup_longitude": [pickup_longitude],
        "pickup_latitude": [pickup_latitude],
        "dropoff_longitude": [dropoff_longitude],
        "dropoff_latitude": [dropoff_latitude],
        "passenger_count": [passenger_count]})

    print("X_test.dtypes")
    print(X.dtypes)

    # TODO: get model from GCP

    # pipeline = get_model_from_gcp()
    pipeline = joblib.load('model.joblib')

    # make prediction
    results = pipeline.predict(X)

    # convert response from numpy to python type
    pred = float(results[0])

    return dict(
        prediction=pred)


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)







