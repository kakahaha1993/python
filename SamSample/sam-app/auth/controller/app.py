import awsgi
from flask import Flask
from service.AuthService import AuthService

app = Flask(__name__)

@app.route('/auth/login', methods=['GET'])
def login():
    authService = AuthService()
    return authService.login()
    

@app.route('/auth/logout', methods=['GET'])
def logout():
    authService = AuthService()
    return authService.logout()

def lambda_handler(event, context):
    return awsgi.response(app, event, context)
