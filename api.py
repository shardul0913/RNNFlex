from flask import Flask, redirect, url_for, request, render_template, jsonify
import tensorflow as tf
from flask_restful import Api, Resource

app = Flask(__name__)
api = Api(app)
print('Visit http://127.0.0.1:5000')

class get_data(Resource):

    def get(self):
        return {"simple":"get"}

    def post(self):
        if (request.method == 'POST'):
            json_in = request.get_json()
            return jsonify({"your":json_in}, 201)
        else:
            return jsonify({"about":"Hello"})

class get_model(Resource):
    def get(self,k):
        if k==1:
            return jsonify({"you chose":"Traditional CNN"})
        elif k==2:
            return jsonify({"you chose": "VGG 16"})
        else:
            return jsonify({"error": "please choose between 1 or 2"}, 500)

api.add_resource(get_data,"/")
api.add_resource(get_model,"/model/<int:k>")

if __name__ == '__main__':
    app.run(port=5000, debug=True)
