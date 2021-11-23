from flask import jsonify

class AuthService:
    def login(self):
        return jsonify({'users': "login"})

    def logout(self):
        return jsonify({'users': "logout"})