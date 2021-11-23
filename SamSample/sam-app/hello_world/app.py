import awsgi
from flask import Flask

app = Flask(__name__)

@app.route('/')
def index():
    return "hello world"

@app.route('/login')
def login():
    return "hello login"

def lambda_handler(event, context):
    return awsgi.response(app, event, context)
